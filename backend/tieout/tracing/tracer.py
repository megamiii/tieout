"""In-process tracing: a tree of spans per agent run or workflow.

The active tracer lives in a context variable, so instrumented code anywhere
(the LLM client, tools, search, validation) records spans with

    with span("tool", "search_filings", input={...}) as s:
        ...
        s.output = ...

and does nothing when no tracer is active (unit tests, evals). FastAPI runs
each request in its own thread with its own context, so concurrent requests
never share a trace.
"""

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

SpanType = Literal[
    "agent", "step", "llm", "tool", "retrieval", "validation", "workflow", "adjudication", "explain"
]


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    seq: int
    type: SpanType
    name: str
    started_at: datetime
    start_ms: float  # offset from the trace start, for the timeline
    duration_ms: float = 0.0
    input: Any = None
    output: Any = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self, kind: str, name: str) -> None:
        self.trace_id = str(uuid.uuid4())
        self.kind = kind
        self.name = name
        self.started_at = datetime.now(UTC)
        self._t0 = time.perf_counter()
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self.llm_messages_seen = 0  # for recording only new messages per LLM call

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @property
    def duration_ms(self) -> float:
        return self.now_ms()

    def open(self, type_: SpanType, name: str, input: Any = None) -> Span:
        s = Span(
            span_id=uuid.uuid4().hex[:16],
            parent_id=self._stack[-1].span_id if self._stack else None,
            seq=len(self.spans),
            type=type_,
            name=name,
            started_at=datetime.now(UTC),
            start_ms=self.now_ms(),
            input=input,
        )
        self.spans.append(s)
        self._stack.append(s)
        return s

    def close(self, s: Span) -> None:
        s.duration_ms = self.now_ms() - s.start_ms
        if self._stack and self._stack[-1] is s:
            self._stack.pop()
        elif s in self._stack:  # closed out of order (an exception unwound): drop it and above
            del self._stack[self._stack.index(s) :]

    def totals(self) -> dict[str, Any]:
        llm = [s for s in self.spans if s.type == "llm"]
        return {
            "llm_calls": len(llm),
            "llm_requests": sum(1 for s in llm if not s.cached),
            "cached": sum(1 for s in llm if s.cached),
            "prompt_tokens": sum(s.prompt_tokens for s in llm if not s.cached),
            "completion_tokens": sum(s.completion_tokens for s in llm if not s.cached),
            "tool_calls": sum(1 for s in self.spans if s.type == "tool"),
            "errors": sum(1 for s in self.spans if s.error),
        }


_current: ContextVar[Tracer | None] = ContextVar("tieout_tracer", default=None)


def current_tracer() -> Tracer | None:
    return _current.get()


@contextmanager
def tracing(kind: str, name: str) -> Iterator[Tracer]:
    """Activate a new tracer for the duration of the block."""
    tracer = Tracer(kind, name)
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)


class _NullSpan(Span):
    """Absorbs attribute writes when tracing is off."""

    def __init__(self) -> None:
        super().__init__("", None, 0, "step", "", datetime.now(UTC), 0.0)


@contextmanager
def span(type_: SpanType, name: str, input: Any = None) -> Iterator[Span]:
    tracer = _current.get()
    if tracer is None:
        yield _NullSpan()
        return
    s = tracer.open(type_, name, input)
    try:
        yield s
    except BaseException as err:
        s.error = f"{type(err).__name__}: {err}"[:1000]
        raise
    finally:
        tracer.close(s)


def annotate(**meta: Any) -> None:
    """Add metadata to the innermost open span, if tracing is on."""
    tracer = _current.get()
    if tracer is not None and tracer._stack:
        tracer._stack[-1].meta.update(meta)
