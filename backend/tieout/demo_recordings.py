"""Build `recordings/demo/`: the only data the public demo's LLM and tools may use.

A trace names every recording file its run used (`request_hash` on LLM
spans, `recording_hash` on tool spans), so a demo recording is assembled by
replaying a run that is already recorded (an eval run, or a stored workflow)
and copying exactly those files. Each copied question is then replayed from
the demo directory alone, which proves the copy is complete.
"""

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tieout.agent.factory import build_agent
from tieout.api.demo import DemoQuestion, questions_path
from tieout.config import REPO_ROOT, Settings
from tieout.financebench.selection import CompanyConfig
from tieout.tracing.tracer import Span, tracing

CURATED_PATH = REPO_ROOT / "config" / "demo_questions.json"


class CuratedQuestion(BaseModel):
    id: str
    eval_item: str  # id in evals/sets/*.jsonl
    source: str  # recording directory the eval run was recorded in
    shows: str


@dataclass
class Dependencies:
    llm: set[str]
    tools: set[str]


def dependencies(spans: Iterable[Span] | Iterable[dict[str, Any]]) -> Dependencies:
    """Recording files a traced run used, from live spans or stored ones."""
    deps = Dependencies(set(), set())
    for s in spans:
        type_, meta = (s["type"], s["meta"]) if isinstance(s, dict) else (s.type, s.meta)
        if type_ == "llm" and meta.get("request_hash"):
            deps.llm.add(meta["request_hash"])
        if type_ == "tool" and meta.get("recording_hash"):
            deps.tools.add(meta["recording_hash"])
    return deps


def copy_files(hashes: Iterable[str], src: Path, dst: Path) -> int:
    """Copy `<src>/<h[:2]>/<h>.json` files; returns how many were new."""
    new = 0
    for h in sorted(hashes):
        rel = Path(h[:2]) / f"{h}.json"
        if not (src / rel).exists():
            raise FileNotFoundError(f"{src / rel} is missing; was the run recorded?")
        target = dst / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, target)
            new += 1
    return new


def _ask(settings: Settings, companies: list[CompanyConfig], recording: Path, question: str) -> Any:
    # The same construction as the API's demo agent, so request hashes match.
    agent = build_agent(settings, companies, offline=True, recording_dir=recording)
    with tracing("agent", question[:200]) as tracer:
        result = agent.ask(question)
    return result, tracer


def build_question(
    settings: Settings,
    companies: list[CompanyConfig],
    curated: CuratedQuestion,
    question: str,
    demo_dir: Path,
) -> dict[str, Any]:
    source = REPO_ROOT / curated.source
    original, tracer = _ask(settings, companies, source, question)
    deps = dependencies(tracer.spans)
    copied = copy_files(deps.llm, source / "llm", demo_dir / "llm")
    copied += copy_files(deps.tools, source / "tools", demo_dir / "tools")
    replay, _ = _ask(settings, companies, demo_dir, question)
    if replay.final.answer != original.final.answer:
        raise RuntimeError(f"{curated.id}: the demo replay differs from the source run")
    return {
        "id": curated.id,
        "status": replay.final.status,
        "llm_calls": len(deps.llm),
        "tool_calls": len(deps.tools),
        "files_copied": copied,
    }


def write_questions(demo_dir: Path, questions: list[DemoQuestion]) -> None:
    path = questions_path(demo_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [q.model_dump() for q in questions]
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _comparable(report: dict[str, Any]) -> str:
    """A workflow report without its wall-clock timing."""
    summary = {k: v for k, v in report["summary"].items() if k != "seconds"}
    return json.dumps({"summary": summary, "items": report["items"]}, default=str, sort_keys=True)


def rebuild_workflow(
    session: Any,
    settings: Settings,
    embedder: Any,
    kind: str,
    ticker: str,
    year: int,
    demo_dir: Path,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Re-run a stored workflow from the local LLM cache (offline: no requests),
    copy the responses it used into the demo recording, and verify the demo
    recording alone reproduces it."""
    from tieout.llm.client import make_llm_client
    from tieout.tracing.tracer import span
    from tieout.workflows.service import run_analytical_review, run_tieout

    run = run_tieout if kind == "tieout" else run_analytical_review
    name = f"{kind} {ticker} FY{year}"
    cached = make_llm_client(settings, offline=True)
    with tracing(kind, name) as tracer, span("workflow", name):
        report = run(session, embedder, cached, ticker, year)
    deps = dependencies(tracer.spans)
    copied = copy_files(deps.llm, settings.llm_cache_dir, demo_dir / "llm")
    demo_llm = make_llm_client(settings, offline=True, cache_dir=demo_dir / "llm")
    replay = run(session, embedder, demo_llm, ticker, year)
    if _comparable(replay) != _comparable(report):
        raise RuntimeError(f"{name}: the demo replay differs from the cached run")
    info = {"workflow": name, "llm_calls": len(deps.llm), "files_copied": copied}
    return report, tracer, info
