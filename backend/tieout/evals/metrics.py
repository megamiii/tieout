"""Aggregate eval results into metrics and JSON + Markdown reports.

Definitions (also in the README):
- accuracy: correct / all items (needs_review and errors count as not correct)
- answered rate: share of items with status "answered"
- accuracy when answered: correct / answered
- citation precision: citations that support the answer / all citations
- abstention precision: P(the answer was wrong | it went to review)
- abstention recall: P(it went to review | the answer was wrong)
- usage: LLM requests and tokens per item (free tier: no dollar cost)
- latency: p50 / p95 of measured live runs (LLM time + tool time)
"""

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tieout.evals.grading import citation_supports
from tieout.evals.runner import ARCHIVED, EXPERIMENTS, load_result
from tieout.evals.sets import EvalItem

LABELS: dict[str, str] = {str(k): v for k, v in EXPERIMENTS.items()} | ARCHIVED


@dataclass
class Metrics:
    experiment: str
    eval_set: str
    n: int
    graded: int
    accuracy: float | None
    answered_rate: float | None
    accuracy_answered: float | None
    numeric_accuracy: float | None
    text_accuracy: float | None
    citation_precision: float | None
    citations: int
    abstention_precision: float | None
    abstention_recall: float | None
    requests_per_item: float | None
    tokens_per_item: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    errors: int


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def compute(experiment: str, eval_set: str, items: list[EvalItem]) -> Metrics:
    rows = [(i, load_result(experiment, i.id)) for i in items]
    done = [(i, r) for i, r in rows if r is not None]
    graded = [(i, r) for i, r in done if r.get("grade") is not None]
    errors = sum(1 for _, r in done if r.get("error"))
    correct = [r["grade"]["correct"] for _, r in graded]
    answered = [(i, r) for i, r in graded if r.get("status") == "answered"]
    review = [(i, r) for i, r in graded if r.get("status") == "needs_review"]
    wrong = [(i, r) for i, r in graded if not r["grade"]["correct"]]

    def acc(kind: str) -> float | None:
        sub = [r["grade"]["correct"] for i, r in graded if i.answer_kind == kind]
        return _ratio(sum(sub), len(sub))

    cits = [citation_supports(i, c) for i, r in graded for c in r.get("citations", [])]
    live = [r for _, r in done if not r.get("error") and r.get("latency_ms") is not None]
    usage_rows = [r for _, r in done if not r.get("error")]
    n_eval = len(graded) + errors
    return Metrics(
        experiment=experiment,
        eval_set=eval_set,
        n=len(items),
        graded=len(graded),
        accuracy=_ratio(sum(correct), n_eval),
        answered_rate=_ratio(len(answered), len(graded)),
        accuracy_answered=_ratio(sum(r["grade"]["correct"] for _, r in answered), len(answered)),
        numeric_accuracy=acc("numeric"),
        text_accuracy=acc("text"),
        citation_precision=_ratio(sum(cits), len(cits)),
        citations=len(cits),
        abstention_precision=_ratio(
            sum(1 for _, r in review if not r["grade"]["correct"]), len(review)
        ),
        abstention_recall=_ratio(
            sum(1 for _, r in wrong if r.get("status") == "needs_review"), len(wrong)
        ),
        requests_per_item=statistics.mean(r["requests"] for r in usage_rows)
        if usage_rows
        else None,
        tokens_per_item=statistics.mean(
            r["prompt_tokens"] + r["completion_tokens"] for r in usage_rows
        )
        if usage_rows
        else None,
        latency_p50_s=(
            p / 1000
            if (p := _percentile([r["latency_ms"] for r in live], 0.5)) is not None
            else None
        ),
        latency_p95_s=(
            p / 1000
            if (p := _percentile([r["latency_ms"] for r in live], 0.95)) is not None
            else None
        ),
        errors=errors,
    )


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def num(x: float | None, fmt: str = "{:,.0f}") -> str:
    return "-" if x is None else fmt.format(x)


def write_report(all_metrics: list[Metrics], out_dir: Path, meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(
        json.dumps({"meta": meta, "metrics": [asdict(m) for m in all_metrics]}, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Eval report",
        "",
        "Run metadata: " + ", ".join(f"{k}={v}" for k, v in meta.items()),
        "",
    ]
    for eval_set in sorted({m.eval_set for m in all_metrics}):
        lines += [
            f"## {eval_set}",
            "",
            "| Configuration | graded | accuracy | answered | acc. when answered | numeric | text "
            "| citation precision | abstention P / R | requests/q | tokens/q | latency p50 / p95 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m in all_metrics:
            if m.eval_set != eval_set:
                continue
            lines.append(
                f"| {LABELS.get(m.experiment, m.experiment)} | {m.graded}/{m.n} "
                f"| {pct(m.accuracy)} | {pct(m.answered_rate)} | {pct(m.accuracy_answered)} "
                f"| {pct(m.numeric_accuracy)} | {pct(m.text_accuracy)} "
                f"| {pct(m.citation_precision)} ({m.citations}) "
                f"| {pct(m.abstention_precision)} / {pct(m.abstention_recall)} "
                f"| {num(m.requests_per_item, '{:.1f}')} | {num(m.tokens_per_item)} "
                f"| {num(m.latency_p50_s, '{:.1f}')} s / {num(m.latency_p95_s, '{:.1f}')} s |"
            )
        lines.append("")
    path = out_dir / "eval.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
