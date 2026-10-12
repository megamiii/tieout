"""Persist agent runs and reviewer decisions; build the views the UI shows."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from tieout.agent.loop import AgentResult
from tieout.db.models import (
    AgentRun,
    ChunkRow,
    Company,
    Filing,
    ReviewDecision,
    XbrlFactRow,
)
from tieout.evals.sets import EvalItem
from tieout.highlight import highlight_all
from tieout.tracing.store import trace_for

Decision = Literal["accept", "override"]
QueueFilter = Literal["pending", "reviewed", "all"]


def _now() -> datetime:
    return datetime.now(UTC)


def trace_of(result: AgentResult) -> list[dict[str, Any]]:
    """A compact, JSON-friendly trace of an agent run (full tracing: milestone 8)."""
    steps = []
    for s in result.steps:
        r = s.response
        steps.append(
            {
                "step": s.index,
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
                "latency_ms": round(r.latency_ms, 1),
                "cached": r.cached,
                "forced_submit": s.forced_submit,
                "nudged": s.nudged,
                "tools": [
                    {
                        "name": o.name,
                        "arguments": o.raw_arguments,
                        "error": o.error,
                        "sources": [e.source_id for e in o.evidence],
                        "latency_ms": round(o.latency_ms, 1),
                    }
                    for o in s.tool_outcomes
                ],
            }
        )
    return steps


def save_live_run(session: Session, result: AgentResult, model: str, source: str = "live") -> str:
    f = result.final
    run_id = str(uuid.uuid4())
    responses = [s.response for s in result.steps]
    citations = []
    for c in f.citations:
        ev = result.evidence.sources.get(c["source_id"])
        citations.append({**c, "values": [str(v) for v in ev.values] if ev else []})
    session.add(
        AgentRun(
            id=run_id,
            created_at=_now(),
            source=source,
            question=result.question,
            status=f.status,
            answer=f.answer,
            confidence=f.confidence,
            rationale=f.rationale,
            citations=citations,
            review_reasons=f.review_reasons,
            warnings=f.warnings,
            stop_reason=result.stop_reason,
            model=model,
            requests=len(responses),
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            latency_ms=sum(r.latency_ms for r in responses),
            trace=trace_of(result),
            eval_item_id=None,
        )
    )
    return run_id


def import_eval_results(
    session: Session, experiment: str, runs_dir: Path, items: list[EvalItem], model: str
) -> int:
    """Load an experiment's eval results as runs (idempotent: keyed by eval item)."""
    by_id = {i.id: i for i in items}
    n = 0
    for path in sorted((runs_dir / experiment).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        item = by_id.get(r.get("item_id", ""))
        if item is None or r.get("error") or "status" not in r:
            continue
        values = {
            "created_at": _now(),
            "source": f"eval:{experiment}",
            "question": item.question,
            "status": r["status"],
            "answer": r.get("answer"),
            "confidence": r.get("confidence"),
            "rationale": r.get("rationale"),
            "citations": r.get("citations", []),
            "review_reasons": r.get("review_reasons", []),
            "warnings": r.get("warnings", []),
            "stop_reason": r.get("stop_reason", ""),
            "model": model,
            "requests": r.get("requests", 0),
            "prompt_tokens": r.get("prompt_tokens", 0),
            "completion_tokens": r.get("completion_tokens", 0),
            "latency_ms": r.get("latency_ms"),
            "trace": r.get("trace", []),
        }
        key = f"{experiment}:{item.id}"
        session.execute(
            pg_insert(AgentRun)
            .values(id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)), eval_item_id=key, **values)
            .on_conflict_do_update(index_elements=[AgentRun.eval_item_id], set_=values)
        )
        n += 1
    return n


def add_decision(
    session: Session,
    run_id: str,
    decision: Decision,
    note: str,
    reviewer: str,
    override_answer: str | None = None,
) -> ReviewDecision:
    run = session.get(AgentRun, run_id)
    if run is None:
        raise KeyError(run_id)
    if decision == "override" and not (override_answer and override_answer.strip()):
        raise ValueError("an override needs the corrected answer")
    final = override_answer.strip() if decision == "override" and override_answer else run.answer
    row = ReviewDecision(
        run_id=run_id,
        created_at=_now(),
        decision=decision,
        final_answer=final or "",
        note=note,
        reviewer=reviewer,
    )
    session.add(row)
    session.flush()
    return row


def _decision_dict(d: ReviewDecision) -> dict[str, Any]:
    return {
        "id": d.id,
        "created_at": d.created_at.isoformat(),
        "decision": d.decision,
        "final_answer": d.final_answer,
        "note": d.note,
        "reviewer": d.reviewer,
    }


def run_summary(run: AgentRun, latest: ReviewDecision | None) -> dict[str, Any]:
    return {
        "id": run.id,
        "created_at": run.created_at.isoformat(),
        "source": run.source,
        "question": run.question,
        "status": run.status,
        "answer": run.answer,
        "confidence": run.confidence,
        "review_reasons": run.review_reasons,
        "review": _decision_dict(latest) if latest else None,
    }


def list_runs(
    session: Session, queue: QueueFilter = "pending", needs_review_only: bool = True
) -> list[dict[str, Any]]:
    latest_id = (
        select(ReviewDecision.run_id, func.max(ReviewDecision.id).label("latest"))
        .group_by(ReviewDecision.run_id)
        .subquery()
    )
    stmt = (
        select(AgentRun, ReviewDecision)
        .outerjoin(latest_id, latest_id.c.run_id == AgentRun.id)
        .outerjoin(ReviewDecision, ReviewDecision.id == latest_id.c.latest)
        .order_by(AgentRun.created_at.desc(), AgentRun.id)
    )
    if needs_review_only:
        stmt = stmt.where(AgentRun.status == "needs_review")
    if queue == "pending":
        stmt = stmt.where(latest_id.c.latest.is_(None))
    elif queue == "reviewed":
        stmt = stmt.where(latest_id.c.latest.is_not(None))
    return [run_summary(run, decision) for run, decision in session.execute(stmt)]


def _source_detail(session: Session, source_id: str, quotes: list[str]) -> dict[str, Any]:
    if source_id.startswith("ch_"):
        row = session.execute(
            select(ChunkRow, Company.ticker, Company.name, Filing.doc_name, Filing.edgar_url)
            .join(Filing, Filing.accession == ChunkRow.accession)
            .join(Company, Company.cik == ChunkRow.cik)
            .where(ChunkRow.chunk_id == source_id)
        ).first()
        if row is None:
            return {"found": False}
        chunk, ticker, name, doc_name, edgar_url = row
        segments, kinds = highlight_all(chunk.text, quotes)
        return {
            "found": True,
            "kind": "chunk",
            "ticker": ticker,
            "company": name,
            "fiscal_year": chunk.fiscal_year,
            "section": chunk.section,
            "section_title": chunk.section_title,
            "page": chunk.page,
            "doc_name": doc_name,
            "edgar_url": edgar_url,
            "segments": segments,
            "matches": kinds,  # per quote: "quote" | "figures" | "none"
        }
    fact = session.get(XbrlFactRow, source_id)
    if fact is None:
        return {"found": False}
    company = session.get(Company, fact.cik)
    return {
        "found": True,
        "kind": "fact",
        "ticker": company.ticker if company else None,
        "concept": fact.concept,
        "label": fact.label,
        "unit": fact.unit,
        "value": str(fact.value),
        "period_start": fact.period_start.isoformat() if fact.period_start else None,
        "period_end": fact.period_end.isoformat(),
        "accession": fact.accession,
        "filed": fact.filed.isoformat(),
    }


def grouped_citations(session: Session, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per cited source, with all of its quotes, in first-cited order."""
    quotes: dict[str, list[str]] = {}
    for c in citations:
        q = quotes.setdefault(c["source_id"], [])
        if c.get("quote") and c["quote"] not in q:
            q.append(c["quote"])
    return [
        {"source_id": sid, "quotes": qs, **_source_detail(session, sid, qs)}
        for sid, qs in quotes.items()
    ]


def run_detail(session: Session, run_id: str) -> dict[str, Any] | None:
    run = session.get(AgentRun, run_id)
    if run is None:
        return None
    decisions = list(
        session.scalars(
            select(ReviewDecision)
            .where(ReviewDecision.run_id == run_id)
            .order_by(ReviewDecision.id)
        )
    )
    detail = run_summary(run, decisions[-1] if decisions else None)
    detail.update(
        {
            "rationale": run.rationale,
            "warnings": run.warnings,
            "stop_reason": run.stop_reason,
            "model": run.model,
            "usage": {
                "requests": run.requests,
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "latency_ms": run.latency_ms,
            },
            "citations": grouped_citations(session, run.citations),
            "trace": run.trace,
            "decisions": [_decision_dict(d) for d in decisions],
            "trace_id": trace_for(session, run_id),
        }
    )
    return detail
