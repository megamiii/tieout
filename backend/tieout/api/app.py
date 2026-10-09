"""FastAPI app: ask the agent, review queue, eval dashboard data."""

import json
from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from tieout import runs
from tieout.agent.factory import Agent, build_agent
from tieout.config import REPO_ROOT, get_settings
from tieout.db.session import get_sessionmaker
from tieout.evals.runner import ARCHIVED, EXPERIMENTS
from tieout.financebench.selection import CompanyConfig
from tieout.ingest import load_selected
from tieout.llm.types import LLMError
from tieout.net.retry import RetryableError

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
    return build_agent(get_settings(), companies())


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
    try:
        result = agent().ask(req.question)
    except RetryableError as err:
        raise HTTPException(429, f"The LLM provider is rate-limiting: {err}") from err
    except LLMError as err:
        raise HTTPException(502, f"LLM error: {err}") from err
    run_id = runs.save_live_run(session, result, model=agent().llm.model)
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
def review(run_id: str, req: ReviewRequest, session: SessionDep) -> dict[str, Any]:
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
