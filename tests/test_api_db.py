"""API tests against real Postgres (skipped if unavailable).

Every request runs in one connection whose transaction is rolled back at the
end, so the tests never change the real database. The agent is not called.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from tieout.api.app import app, get_session
from tieout.db.models import AgentRun
from tieout.db.session import get_engine, init_db

pytestmark = pytest.mark.db


@pytest.fixture
def session() -> Iterator[Session]:
    engine = get_engine()
    try:
        init_db(engine)
    except OperationalError:
        pytest.skip("Postgres not running (docker compose up -d)")
    conn = engine.connect()
    trans = conn.begin()
    # Commits inside the API become savepoints, all undone by the rollback below.
    s = Session(bind=conn, join_transaction_mode="create_savepoint")
    try:
        yield s
    finally:
        s.close()
        trans.rollback()
        conn.close()


@pytest.fixture
def client(session: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_session] = lambda: session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def add_run(session: Session, run_id: str, status: str) -> None:
    session.add(
        AgentRun(
            id=run_id,
            created_at=datetime(2030, 1, 1, tzinfo=UTC),  # newest first in listings
            source="test",
            question="Q?",
            status=status,
            answer="It was $1,577 million.",
            confidence=0.5,
            rationale="r",
            citations=[{"source_id": "xf_0000000000000000", "quote": "1577000000"}],
            review_reasons=["confidence 0.50 is below 0.70"],
            warnings=[],
            stop_reason="submitted",
            model="m",
            requests=3,
            prompt_tokens=100,
            completion_tokens=10,
            latency_ms=1.0,
            trace=[],
            eval_item_id=None,
        )
    )
    session.flush()


def test_review_queue_accept_and_override(client: TestClient, session: Session) -> None:
    add_run(session, "test-run-1", "needs_review")
    add_run(session, "test-run-2", "answered")

    pending = client.get("/api/runs", params={"queue": "pending"}).json()
    ids = [r["id"] for r in pending]
    assert "test-run-1" in ids and "test-run-2" not in ids  # only needs_review runs

    r = client.post(
        "/api/runs/test-run-1/review",
        json={"decision": "accept", "reviewer": "auditor", "note": "checked the 10-K"},
    )
    assert r.status_code == 200
    assert r.json()["review"]["final_answer"] == "It was $1,577 million."

    r = client.post(
        "/api/runs/test-run-1/review",
        json={"decision": "override", "reviewer": "lead", "override_answer": "$1,577M (FY2018)"},
    )
    detail = r.json()
    assert detail["review"]["decision"] == "override"
    assert [d["decision"] for d in detail["decisions"]] == ["accept", "override"]  # history kept

    pending = [r["id"] for r in client.get("/api/runs", params={"queue": "pending"}).json()]
    reviewed = [r["id"] for r in client.get("/api/runs", params={"queue": "reviewed"}).json()]
    assert "test-run-1" not in pending and "test-run-1" in reviewed


def test_review_validation(client: TestClient, session: Session) -> None:
    add_run(session, "test-run-3", "needs_review")
    missing = {"decision": "override", "reviewer": "x"}
    assert client.post("/api/runs/test-run-3/review", json=missing).status_code == 422
    assert (
        client.post("/api/runs/nope/review", json={**missing, "decision": "accept"}).status_code
        == 404
    )
    assert (
        client.post(
            "/api/runs/test-run-3/review", json={"decision": "maybe", "reviewer": "x"}
        ).status_code
        == 422
    )


def test_unknown_citation_source_is_reported_not_raised(
    client: TestClient, session: Session
) -> None:
    add_run(session, "test-run-4", "needs_review")
    detail = client.get("/api/runs/test-run-4").json()
    assert detail["citations"][0]["found"] is False


def test_ask_persists_the_run(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # The real agent loop with a scripted model and a stub search tool: no LLM, no index.
    from test_agent_loop import SEARCH, ScriptedLLM, StubSearch, resp, submit
    from tieout.agent.factory import Agent
    from tieout.agent.finance_tools import CalculateTool, SubmitAnswerTool
    from tieout.agent.loop import AgentLimits
    from tieout.api import app as app_module

    llm = ScriptedLLM([resp(SEARCH), resp(submit("3M's FY2018 capex was $1,577 million."))])
    stub = Agent(llm, [StubSearch(), CalculateTool(), SubmitAnswerTool()], "sys", AgentLimits())
    monkeypatch.setattr(app_module, "agent", lambda: stub)

    r = client.post("/api/ask", json={"question": "What was 3M's FY2018 capex?"})
    assert r.status_code == 200
    run = r.json()
    assert run["status"] == "answered" and run["source"] == "live"
    assert run["usage"]["requests"] == 2
    assert [s["tools"][0]["name"] for s in run["trace"]] == ["search_filings", "submit_answer"]
    assert client.get(f"/api/runs/{run['id']}").json()["answer"] == run["answer"]


def test_workflow_endpoints_store_and_list_runs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tieout.api import app as app_module

    report = {
        "kind": "tieout",
        "ticker": "MMM",
        "fiscal_year": 2018,
        "doc_name": "3M_2018_10K",
        "summary": {"claims": 1, "matched": 1, "mismatched": 0, "unmatched": 0},
        "items": [{"status": "matched", "amount": "$1,577 million"}],
    }

    class StubAgent:
        llm = type("L", (), {"model": "stub"})()

    monkeypatch.setattr(app_module, "agent", lambda: StubAgent())
    monkeypatch.setattr(app_module, "embedder", lambda: None)
    monkeypatch.setattr(app_module, "run_tieout", lambda *a, **k: report)

    run = client.post("/api/workflows/tieout", json={"ticker": "MMM", "fiscal_year": 2018}).json()
    assert run["summary"]["matched"] == 1 and run["model"] == "stub"
    listed = client.get("/api/workflows", params={"kind": "tieout"}).json()
    assert run["id"] in [r["id"] for r in listed]
    assert all(r["items"] is None for r in listed)  # list view stays light
    assert (
        client.get(f"/api/workflows/{run['id']}").json()["items"][0]["amount"] == "$1,577 million"
    )
    assert client.get("/api/workflows/nope").status_code == 404


def test_trace_is_saved_with_the_run_and_served(client: TestClient, session: Session) -> None:
    from tieout.tracing.store import save_trace
    from tieout.tracing.tracer import span, tracing

    add_run(session, "test-run-5", "answered")
    with tracing("agent", "Q?") as tr, span("agent", "agent run"), span("tool", "search") as s:
        s.output = {"hits": 1}
    trace_id = save_trace(session, tr, "test-run-5", "m")
    session.flush()

    detail = client.get("/api/runs/test-run-5").json()
    assert detail["trace_id"] == trace_id
    trace = client.get(f"/api/traces/{trace_id}").json()
    assert [s["type"] for s in trace["spans"]] == ["agent", "tool"]
    assert trace["spans"][1]["parent_id"] == trace["spans"][0]["span_id"]
    assert trace["spans"][1]["output"] == {"hits": 1}
    assert client.get("/api/traces/nope").status_code == 404


# --- demo mode -------------------------------------------------------------------


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from tieout.api import app as app_module
    from tieout.config import Settings

    settings = Settings(demo_mode=True, demo_recording_dir=tmp_path)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    return tmp_path


def test_demo_config_lists_the_recorded_questions(client: TestClient, demo: Path) -> None:
    (demo / "questions.json").write_text(
        json.dumps([{"id": "capex", "question": "What was 3M's capex?", "shows": "XBRL"}]),
        encoding="utf-8",
    )
    config = client.get("/api/config").json()
    assert config["demo_mode"] is True
    assert [q["id"] for q in config["demo_questions"]] == ["capex"]


def test_demo_question_is_replayed_once_then_served_from_the_database(
    client: TestClient, demo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_agent_loop import SEARCH, ScriptedLLM, StubSearch, resp, submit
    from tieout.agent.factory import Agent
    from tieout.agent.finance_tools import SubmitAnswerTool
    from tieout.agent.loop import AgentLimits
    from tieout.api import app as app_module

    # Exactly one run's worth of responses: a second agent run would fail.
    llm = ScriptedLLM([resp(SEARCH), resp(submit("3M's FY2018 capex was $1,577 million."))])
    stub = Agent(llm, [StubSearch(), SubmitAnswerTool()], "sys", AgentLimits())
    monkeypatch.setattr(app_module, "agent", lambda: stub)

    q = {"question": "What was 3M's FY2018 capex? (demo test)"}
    first = client.post("/api/ask", json=q).json()
    assert first["source"] == "demo"
    second = client.post("/api/ask", json=q).json()
    assert second["id"] == first["id"]


def test_demo_refuses_what_was_not_recorded(
    client: TestClient, demo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tieout.api import app as app_module
    from tieout.llm.types import LLMCacheMiss

    class Unrecorded:
        llm = type("L", (), {"model": "stub"})()

        def ask(self, question: str) -> None:
            raise LLMCacheMiss("request abc not in recorded cache")

    monkeypatch.setattr(app_module, "agent", lambda: Unrecorded())
    r = client.post("/api/ask", json={"question": "Something nobody recorded?"})
    assert r.status_code == 403
    assert "only replays recorded runs" in r.json()["detail"]


def test_demo_workflow_is_stored_once(
    client: TestClient, demo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tieout.api import app as app_module

    calls: list[str] = []

    def fake_tieout(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append("run")
        return {
            "kind": "tieout",
            "ticker": "ZZZ",
            "fiscal_year": 2099,
            "doc_name": "ZZZ_2099_10K",
            "summary": {"claims": 0},
            "items": [],
        }

    class StubAgent:
        llm = type("L", (), {"model": "stub"})()

    monkeypatch.setattr(app_module, "agent", lambda: StubAgent())
    monkeypatch.setattr(app_module, "embedder", lambda: None)
    monkeypatch.setattr(app_module, "run_tieout", fake_tieout)
    body = {"ticker": "ZZZ", "fiscal_year": 2099}
    first = client.post("/api/workflows/tieout", json=body).json()
    second = client.post("/api/workflows/tieout", json=body).json()
    assert second["id"] == first["id"] and calls == ["run"]


def test_demo_limits_review_writes_per_client(
    client: TestClient, session: Session, demo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tieout.api import app as app_module
    from tieout.api.demo import WriteLimiter

    limiter = WriteLimiter(1)
    monkeypatch.setattr(app_module, "write_limiter", lambda: limiter)
    add_run(session, "test-run-6", "needs_review")
    body = {"decision": "accept", "reviewer": "demo", "note": ""}
    assert client.post("/api/runs/test-run-6/review", json=body).status_code == 200
    assert client.post("/api/runs/test-run-6/review", json=body).status_code == 429
