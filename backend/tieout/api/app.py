"""FastAPI app: ask the agent, review queue, eval dashboard data."""

import json
from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from tieout import runs
from tieout.agent.factory import Agent, build_agent
from tieout.agent.recording import ToolReplayMiss
from tieout.api.demo import NOT_RECORDED, WriteLimiter, client_key, load_demo_questions
from tieout.config import REPO_ROOT, get_settings
from tieout.db.models import AgentRun, WorkflowRun
from tieout.db.session import get_sessionmaker
from tieout.evals.runner import ARCHIVED, EXPERIMENTS
from tieout.financebench.selection import CompanyConfig
from tieout.ingest import load_selected
from tieout.llm.types import LLMCacheMiss, LLMError
from tieout.net.retry import RetryableError
from tieout.retrieval.embeddings import Embedder, get_embedder
from tieout.tracing.store import get_trace, save_trace, trace_for
from tieout.tracing.tracer import span, tracing
from tieout.workflows.service import (
    run_analytical_review,
    run_tieout,
    save_workflow,
    workflow_dict,
)

FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"
EVAL_REPORT = REPO_ROOT / "evals" / "reports" / "eval.json"

app = FastAPI(title="tieout", version="0.1.0")


def get_session() -> Iterator[Session]:
    with get_sessionmaker()() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


@lru_cache
def companies() -> list[CompanyConfig]:
    return load_selected()


@lru_cache
def agent() -> Agent:
    settings = get_settings()
    if settings.demo_mode:
        # Replay only: LLM responses and tool outputs come from the demo recording.
        return build_agent(settings, companies(), offline=True, recording_dir=settings.demo_path)
    return build_agent(settings, companies())


@lru_cache
def write_limiter() -> WriteLimiter:
    return WriteLimiter(get_settings().demo_writes_per_hour)


NOT_RECORDED_ERRORS = (LLMCacheMiss, ToolReplayMiss)


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)


class ReviewRequest(BaseModel):
    decision: Literal["accept", "override"]
    note: str = Field(default="", max_length=2000)
    reviewer: str = Field(min_length=1, max_length=100)
    override_answer: str | None = Field(default=None, max_length=4000)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def ui_config() -> dict[str, Any]:
    settings = get_settings()
    return {
        "demo_mode": settings.demo_mode,
        "demo_questions": [q.model_dump() for q in load_demo_questions(settings.demo_path)]
        if settings.demo_mode
        else [],
        "model": settings.llm_model,
    }


@app.get("/api/companies")
def list_companies() -> list[dict[str, Any]]:
    return [
        {
            "ticker": c.ticker,
            "name": c.name,
            "sector": c.sector,
            "fiscal_years": [f.fiscal_year for f in c.filings],
        }
        for c in companies()
    ]


@app.post("/api/ask")
def ask(req: AskRequest, session: SessionDep) -> dict[str, Any]:
    demo = get_settings().demo_mode
    if demo:
        # A recorded question replays identically, so it is stored once and
        # served from then on: public clicks don't grow the database.
        existing = session.scalar(
            select(AgentRun.id)
            .where(AgentRun.source == "demo", AgentRun.question == req.question)
            .order_by(AgentRun.created_at.desc())
            .limit(1)
        )
        if existing:
            detail = runs.run_detail(session, existing)
            assert detail is not None
            return detail
    try:
        with tracing("agent", req.question[:200]) as tracer:
            result = agent().ask(req.question)
    except NOT_RECORDED_ERRORS as err:
        raise HTTPException(403, NOT_RECORDED) from err
    except RetryableError as err:
        raise HTTPException(429, f"The LLM provider is rate-limiting: {err}") from err
    except LLMError as err:
        raise HTTPException(502, f"LLM error: {err}") from err
    run_id = runs.save_live_run(
        session, result, model=agent().llm.model, source="demo" if demo else "live"
    )
    save_trace(session, tracer, run_id, agent().llm.model)
    session.commit()
    detail = runs.run_detail(session, run_id)
    assert detail is not None
    return detail


@app.get("/api/runs")
def list_runs(
    session: SessionDep,
    queue: Literal["pending", "reviewed", "all"] = "pending",
    needs_review_only: bool = True,
) -> list[dict[str, Any]]:
    return runs.list_runs(session, queue, needs_review_only)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, session: SessionDep) -> dict[str, Any]:
    detail = runs.run_detail(session, run_id)
    if detail is None:
        raise HTTPException(404, "run not found")
    return detail


@app.post("/api/runs/{run_id}/review")
def review(
    run_id: str, req: ReviewRequest, request: Request, session: SessionDep
) -> dict[str, Any]:
    if get_settings().demo_mode and not write_limiter().allow(client_key(request)):
        raise HTTPException(429, "Too many review decisions from this address; try later.")
    try:
        runs.add_decision(
            session, run_id, req.decision, req.note, req.reviewer, req.override_answer
        )
    except KeyError as err:
        raise HTTPException(404, "run not found") from err
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    session.commit()
    detail = runs.run_detail(session, run_id)
    assert detail is not None
    return detail


class WorkflowRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    fiscal_year: int = Field(ge=2000, le=2100)


@lru_cache
def embedder() -> Embedder:
    return get_embedder(get_settings().data_path / "models")


def _run_workflow(kind: str, req: WorkflowRequest, session: Session) -> dict[str, Any]:
    llm = agent().llm
    name = f"{kind} {req.ticker.upper()} FY{req.fiscal_year}"
    if get_settings().demo_mode:
        # Same rule as questions: a recorded workflow is stored once, then served.
        existing = session.scalar(
            select(WorkflowRun)
            .where(
                WorkflowRun.kind == kind,
                WorkflowRun.ticker == req.ticker.upper(),
                WorkflowRun.fiscal_year == req.fiscal_year,
            )
            .order_by(WorkflowRun.created_at.desc())
            .limit(1)
        )
        if existing:
            return {**workflow_dict(existing), "trace_id": trace_for(session, existing.id)}
    try:
        with tracing(kind, name) as tracer, span("workflow", name):
            if kind == "tieout":
                report = run_tieout(session, embedder(), llm, req.ticker, req.fiscal_year)
            else:
                report = run_analytical_review(
                    session, embedder(), llm, req.ticker, req.fiscal_year
                )
    except KeyError as err:
        raise HTTPException(404, str(err)) from err
    except NOT_RECORDED_ERRORS as err:
        raise HTTPException(403, NOT_RECORDED) from err
    except RetryableError as err:
        raise HTTPException(429, f"The LLM provider is rate-limiting: {err}") from err
    except LLMError as err:
        raise HTTPException(502, f"LLM error: {err}") from err
    run_id = save_workflow(session, report, llm.model)
    trace_id = save_trace(session, tracer, run_id, llm.model)
    session.commit()
    run = session.get(WorkflowRun, run_id)
    assert run is not None
    return {**workflow_dict(run), "trace_id": trace_id}


@app.post("/api/workflows/tieout")
def tieout(req: WorkflowRequest, session: SessionDep) -> dict[str, Any]:
    return _run_workflow("tieout", req, session)


@app.post("/api/workflows/review")
def analytical_review(req: WorkflowRequest, session: SessionDep) -> dict[str, Any]:
    return _run_workflow("analytical_review", req, session)


@app.get("/api/workflows")
def list_workflows(
    session: SessionDep, kind: Literal["tieout", "analytical_review"] | None = None
) -> list[dict[str, Any]]:
    stmt = select(WorkflowRun).order_by(WorkflowRun.created_at.desc())
    if kind:
        stmt = stmt.where(WorkflowRun.kind == kind)
    return [{**workflow_dict(r), "items": None} for r in session.scalars(stmt)]


@app.get("/api/workflows/{run_id}")
def get_workflow(run_id: str, session: SessionDep) -> dict[str, Any]:
    run = session.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(404, "workflow run not found")
    return {**workflow_dict(run), "trace_id": trace_for(session, run_id)}


@app.get("/api/traces/{trace_id}")
def get_trace_detail(trace_id: str, session: SessionDep) -> dict[str, Any]:
    trace = get_trace(session, trace_id)
    if trace is None:
        raise HTTPException(404, "trace not found")
    return trace


@app.get("/api/eval")
def eval_report() -> dict[str, Any]:
    if not EVAL_REPORT.exists():
        raise HTTPException(404, "no eval report yet; run `tieout eval report`")
    report: dict[str, Any] = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
    report["experiments"] = {**EXPERIMENTS, **ARCHIVED}
    return report


# The built frontend (production); in development Vite serves it with a proxy.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
