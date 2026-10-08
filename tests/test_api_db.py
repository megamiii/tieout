"""API tests against real Postgres (skipped if unavailable).

Every request runs in one connection whose transaction is rolled back at the
end, so the tests never change the real database. The agent is not called.
"""

from collections.abc import Iterator
from datetime import UTC, datetime

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
