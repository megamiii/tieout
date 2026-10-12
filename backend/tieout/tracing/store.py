"""Save traces to Postgres and read them back for the UI."""

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tieout.db.models import SpanRow, TraceRow
from tieout.tracing.tracer import Tracer

MAX_FIELD_CHARS = 20_000  # one span's input or output; long tool outputs are truncated


def json_safe(value: Any) -> Any:
    """Decimals, dates and dataclass-ish objects -> JSON; oversized values truncated."""
    if value is None:
        return None
    text = json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > MAX_FIELD_CHARS:
        return {"truncated": True, "preview": text[:MAX_FIELD_CHARS]}
    return json.loads(text)


def save_trace(session: Session, tracer: Tracer, ref_id: str | None, model: str) -> str:
    session.add(
        TraceRow(
            trace_id=tracer.trace_id,
            kind=tracer.kind,
            name=tracer.name,
            ref_id=ref_id,
            started_at=tracer.started_at,
            duration_ms=round(tracer.duration_ms, 1),
            model=model,
            totals=tracer.totals(),
        )
    )
    session.flush()
    for s in tracer.spans:
        session.add(
            SpanRow(
                trace_id=tracer.trace_id,
                span_id=s.span_id,
                parent_id=s.parent_id,
                seq=s.seq,
                type=s.type,
                name=s.name,
                started_at=s.started_at,
                start_ms=round(s.start_ms, 1),
                duration_ms=round(s.duration_ms, 1),
                input=json_safe(s.input),
                output=json_safe(s.output),
                error=s.error,
                prompt_tokens=s.prompt_tokens,
                completion_tokens=s.completion_tokens,
                cached=s.cached,
                meta=json_safe(s.meta) or {},
            )
        )
    return tracer.trace_id


def trace_for(session: Session, ref_id: str) -> str | None:
    return session.scalar(
        select(TraceRow.trace_id)
        .where(TraceRow.ref_id == ref_id)
        .order_by(TraceRow.started_at.desc())
    )


def get_trace(session: Session, trace_id: str) -> dict[str, Any] | None:
    trace = session.get(TraceRow, trace_id)
    if trace is None:
        return None
    spans = session.scalars(
        select(SpanRow).where(SpanRow.trace_id == trace_id).order_by(SpanRow.seq)
    )
    return {
        "trace_id": trace.trace_id,
        "kind": trace.kind,
        "name": trace.name,
        "ref_id": trace.ref_id,
        "started_at": trace.started_at.isoformat(),
        "duration_ms": trace.duration_ms,
        "model": trace.model,
        "totals": trace.totals,
        "spans": [
            {
                "span_id": s.span_id,
                "parent_id": s.parent_id,
                "seq": s.seq,
                "type": s.type,
                "name": s.name,
                "start_ms": s.start_ms,
                "duration_ms": s.duration_ms,
                "input": s.input,
                "output": s.output,
                "error": s.error,
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
                "cached": s.cached,
                "meta": s.meta,
            }
            for s in spans
        ],
    }
