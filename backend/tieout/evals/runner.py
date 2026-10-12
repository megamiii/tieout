"""Resumable eval runner.

Each (experiment, item) gets one result file in evals/runs/<experiment>/.
Re-running skips items that already have a result, so a run spread over
several days (free-tier quotas) simply continues. LLM and tool calls are
recorded under recordings/evals/<experiment>/, so a finished run replays
offline (no key, database or model) and CI can check the metrics reproduce.

Two phases: answer (agent or baseline), then grade (numeric match, or the
LLM judge for text answers). Each stops cleanly before the day's request
budget is used up or when the provider reports the quota is exhausted.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tieout.agent.factory import build_agent, build_tools
from tieout.agent.loop import FinalAnswer
from tieout.agent.validation import EvidenceLedger
from tieout.config import REPO_ROOT, Settings
from tieout.evals.baseline import run_baseline
from tieout.evals.grading import ChatModel, grade
from tieout.evals.sets import EvalItem
from tieout.financebench.selection import CompanyConfig
from tieout.llm.client import LLMClient, configured_limits, make_llm_client, quota_timezone
from tieout.llm.limits import UsageLedger, quota_day
from tieout.llm.types import ChatResponse, LLMError
from tieout.net.retry import RetryableError

Experiment = Literal["baseline", "agent_hybrid", "agent_full"]
EXPERIMENTS: dict[Experiment, str] = {
    "baseline": "Single prompt, vector RAG only (baseline)",
    "agent_hybrid": "Agent + hybrid RAG",
    "agent_full": "Agent + hybrid RAG + XBRL + calculator",
}
# Earlier versions kept for comparison in the report; not re-runnable with current code.
ARCHIVED: dict[str, str] = {
    "agent_full_v1": "Agent + hybrid RAG + XBRL + calculator, validator v1 (archived)",
}
RUNS_DIR = REPO_ROOT / "evals" / "runs"
RECORDINGS_DIR = REPO_ROOT / "recordings" / "evals"
# Requests one item may need at most (agent steps); stop before the budget can't cover it.
REQUESTS_PER_ITEM = 10


@dataclass
class RunSummary:
    done: int = 0
    skipped: int = 0
    errors: int = 0
    stopped: str | None = None  # why the run stopped early, if it did


def result_path(experiment: str, item_id: str) -> Path:
    return RUNS_DIR / experiment / f"{item_id}.json"


def load_result(experiment: str, item_id: str) -> dict[str, Any] | None:
    path = result_path(experiment, item_id)
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def save_result(experiment: str, item_id: str, data: dict[str, Any]) -> None:
    path = result_path(experiment, item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


class Budget:
    """Stops before the day's request quota for a model is used up."""

    def __init__(self, ledger: UsageLedger, model: str, daily_limit: int | None, reserve: int):
        self.ledger = ledger
        self.model = model
        self.daily_limit = daily_limit
        self.reserve = reserve

    def used_today(self) -> int:
        return int(self.ledger.read().get(self.model, {}).get("requests", 0))

    def exhausted(self) -> bool:
        if not self.daily_limit:
            return False
        return self.used_today() + self.reserve > self.daily_limit


def _record(
    item: EvalItem,
    experiment: str,
    final: FinalAnswer,
    responses: list[ChatResponse],
    ledger: EvidenceLedger,
    tool_ms: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    citations = []
    for c in final.citations:
        ev = ledger.sources.get(c["source_id"])
        # Store what the cited source states; whether it supports the answer is
        # computed at report time, so the rule can change without re-running.
        citations.append({**c, "values": [str(v) for v in ev.values] if ev else []})
    live = [r for r in responses if not r.cached]
    return {
        "item_id": item.id,
        "eval_set": item.eval_set,
        "experiment": experiment,
        "status": final.status,
        "answer": final.answer,
        "confidence": final.confidence,
        "rationale": final.rationale,
        "citations": citations,
        "review_reasons": final.review_reasons,
        "warnings": final.warnings,
        "requests": len(responses),
        "prompt_tokens": sum(r.usage.prompt_tokens for r in responses),
        "completion_tokens": sum(r.usage.completion_tokens for r in responses),
        # Measured on the live run only; replays from the cache take ~0 ms.
        "latency_ms": round(sum(r.latency_ms for r in responses) + tool_ms, 1)
        if len(live) == len(responses)
        else None,
        "grade": None,
        **extra,
    }


def make_answerer(
    experiment: Experiment,
    settings: Settings,
    companies: list[CompanyConfig],
    *,
    offline: bool,
) -> tuple[Callable[[EvalItem], dict[str, Any]], LLMClient]:
    rec = RECORDINGS_DIR / experiment
    llm = make_llm_client(settings, offline=offline, cache_dir=rec / "llm")

    if experiment == "baseline":
        tools = build_tools(
            settings,
            companies,
            search_mode="vector",
            use_xbrl=False,
            recording_dir=rec,
            offline=offline,
        )
        search_tool = next(t for t in tools if t.name == "search_filings")

        def answer_baseline(item: EvalItem) -> dict[str, Any]:
            final, responses, ledger, tool_ms = run_baseline(
                llm, search_tool, companies, item.question
            )
            return _record(
                item,
                experiment,
                final,
                responses,
                ledger,
                tool_ms,
                {"stop_reason": "single_call", "tool_calls": 1, "rejections": 0},
            )

        return answer_baseline, llm

    agent = build_agent(
        settings,
        companies,
        llm=llm,
        search_mode="hybrid",
        use_xbrl=experiment == "agent_full",
        recording_dir=rec,
        offline=offline,
    )

    def answer_agent(item: EvalItem) -> dict[str, Any]:
        result = agent.ask(item.question)
        responses = [s.response for s in result.steps]
        tool_ms = sum(o.latency_ms for s in result.steps for o in s.tool_outcomes)
        return _record(
            item,
            experiment,
            result.final,
            responses,
            result.evidence,
            tool_ms,
            {
                "stop_reason": result.stop_reason,
                "tool_calls": result.tool_calls,
                "rejections": result.rejections,
                "trace": [
                    {
                        "step": s.index,
                        "tokens": s.response.usage.total_tokens,
                        "tools": [
                            {"name": o.name, "arguments": o.raw_arguments, "error": o.error}
                            for o in s.tool_outcomes
                        ],
                    }
                    for s in result.steps
                ],
            },
        )

    return answer_agent, llm


def run_answers(
    experiment: Experiment,
    items: list[EvalItem],
    settings: Settings,
    companies: list[CompanyConfig],
    *,
    offline: bool = False,
    limit: int | None = None,
    retry_errors: bool = False,
    on_item: Callable[[str, dict[str, Any] | None, str], None] | None = None,
) -> RunSummary:
    answer, llm = make_answerer(experiment, settings, companies, offline=offline)
    limits = configured_limits(settings)
    budget = Budget(
        UsageLedger(settings.usage_dir, today=quota_day(quota_timezone(settings))),
        llm.model,
        limits.rpd if limits else None,
        REQUESTS_PER_ITEM,
    )
    summary = RunSummary()
    for item in items:
        existing = load_result(experiment, item.id)
        if existing and not (retry_errors and existing.get("error")):
            summary.skipped += 1
            continue
        if limit is not None and summary.done >= limit:
            summary.stopped = f"reached --limit {limit}"
            break
        if not offline and budget.exhausted():
            summary.stopped = (
                f"daily request budget for {llm.model} nearly used "
                f"({budget.used_today()}/{budget.daily_limit}); resume after the quota resets"
            )
            break
        try:
            data = answer(item)
        except RetryableError as err:
            summary.stopped = f"LLM unavailable after retries (rate limit or network): {err}"
            break
        except LLMError as err:
            data = {
                "item_id": item.id,
                "eval_set": item.eval_set,
                "experiment": experiment,
                "error": str(err)[:500],
                "grade": None,
            }
            summary.errors += 1
        save_result(experiment, item.id, data)
        summary.done += 1
        if on_item:
            on_item(item.id, data, "answered")
    return summary


def make_judge(settings: Settings, *, offline: bool) -> LLMClient:
    judge_settings = settings.model_copy(
        update={
            "llm_model": settings.judge_model,
            "llm_rpm": settings.judge_rpm,
            "llm_rpd": settings.judge_rpd,
        }
    )
    return make_llm_client(
        judge_settings, offline=offline, cache_dir=RECORDINGS_DIR / "judge" / "llm"
    )


def run_grading(
    experiment: Experiment,
    items: list[EvalItem],
    settings: Settings,
    *,
    offline: bool = False,
    regrade: bool = False,
) -> RunSummary:
    judge_llm: ChatModel = make_judge(settings, offline=offline)
    budget = Budget(
        UsageLedger(settings.usage_dir, today=quota_day(quota_timezone(settings))),
        settings.judge_model,
        settings.judge_rpd or None,
        reserve=2,
    )
    summary = RunSummary()
    for item in items:
        data = load_result(experiment, item.id)
        if data is None or data.get("error"):
            continue
        if data.get("grade") is not None and not regrade:
            summary.skipped += 1
            continue
        if item.answer_kind == "text" and not offline and budget.exhausted():
            summary.stopped = "daily request budget for the judge model nearly used"
            break
        try:
            g = grade(item, data.get("answer"), judge_llm)
        except RetryableError as err:
            summary.stopped = f"judge unavailable after retries (rate limit or network): {err}"
            break
        data["grade"] = {
            "correct": g.correct,
            "method": g.method,
            "detail": g.detail,
            "judge_model": settings.judge_model if g.method == "judge" else None,
        }
        save_result(experiment, item.id, data)
        summary.done += 1
    return summary


# --- reproducibility check (CI) -------------------------------------------------

# What must match for a replayed answer to count as reproduced. Latency and
# token counts are excluded: replays are served from the cache.
COMPARED = ("status", "answer", "stop_reason")


@dataclass
class Replay:
    item_id: str
    outcome: Literal["same", "different", "not_recorded", "no_result"]
    detail: str = ""


def verify_replay(
    experiment: Experiment,
    items: list[EvalItem],
    settings: Settings,
    companies: list[CompanyConfig],
) -> list[Replay]:
    """Re-answer every item offline from the recordings and compare with the
    committed result. Nothing is written. A request the recordings don't hold
    (because the code now asks something different) is reported, not fatal."""
    from tieout.agent.recording import ToolReplayMiss
    from tieout.llm.types import LLMCacheMiss

    answer, _ = make_answerer(experiment, settings, companies, offline=True)
    out: list[Replay] = []
    for item in items:
        committed = load_result(experiment, item.id)
        if committed is None or committed.get("error"):
            out.append(Replay(item.id, "no_result"))
            continue
        try:
            fresh = answer(item)
        except (LLMCacheMiss, ToolReplayMiss) as err:
            out.append(Replay(item.id, "not_recorded", str(err)[:200]))
            continue
        diffs = [k for k in COMPARED if fresh.get(k) != committed.get(k)]
        cited = [c["source_id"] for c in fresh.get("citations", [])]
        if cited != [c["source_id"] for c in committed.get("citations", [])]:
            diffs.append("citations")
        out.append(Replay(item.id, "different" if diffs else "same", ", ".join(diffs)))
    return out
