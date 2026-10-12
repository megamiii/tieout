import json
from pathlib import Path
from typing import Any

import pytest

from tieout.demo_static import export_static


class FakeResponse:
    def __init__(self, data: Any) -> None:
        self.data = data

    def json(self) -> Any:
        return self.data

    def raise_for_status(self) -> None:
        pass


class FakeApi:
    """Answers like the demo API, with one question, one run and one workflow."""

    def __init__(self, demo_mode: bool = True) -> None:
        self.demo_mode = demo_mode
        self.calls: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.calls.append(f"GET {url}")
        routes: dict[str, Any] = {
            "/api/config": {
                "demo_mode": self.demo_mode,
                "demo_questions": [{"id": "q", "question": "Q?", "shows": "s"}],
            },
            "/api/companies": [{"ticker": "MMM"}],
            "/api/eval": {"metrics": []},
            "/api/runs?queue=pending": [{"id": "r2"}],
            "/api/runs?queue=reviewed": [],
            "/api/runs?queue=all": [{"id": "r2"}],
            "/api/runs/r1": {"id": "r1", "trace_id": "t1"},
            "/api/runs/r2": {"id": "r2", "trace_id": None},
            "/api/workflows?kind=tieout": [{"id": "w1"}],
            "/api/workflows?kind=analytical_review": [],
            "/api/workflows/w1": {"id": "w1", "items": [], "trace_id": "t2"},
            "/api/traces/t1": {"trace_id": "t1", "spans": []},
            "/api/traces/t2": {"trace_id": "t2", "spans": []},
        }
        return FakeResponse(routes[url])

    def post(self, url: str, *, json: Any) -> FakeResponse:
        self.calls.append(f"POST {url}")
        assert url == "/api/ask" and json == {"question": "Q?"}
        return FakeResponse({"id": "r1"})


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_writes_every_response_the_static_site_reads(tmp_path: Path) -> None:
    out = tmp_path / "demo-data"
    counts = export_static(FakeApi(), out)
    assert read(out / "config.json")["static"] is True
    assert read(out / "asks.json") == {"Q?": "r1"}  # the recorded question -> its run
    for rel in [
        "companies.json",
        "eval.json",
        "runs/queue-pending.json",
        "runs/r1.json",  # asked
        "runs/r2.json",  # listed in the review queue
        "workflows/list-tieout.json",
        "workflows/w1.json",
        "traces/t1.json",  # from the run
        "traces/t2.json",  # from the workflow
    ]:
        assert (out / rel).exists(), rel
    assert counts["traces"] == 2


def test_export_refuses_a_live_api_and_foreign_directories(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not in demo mode"):
        export_static(FakeApi(demo_mode=False), tmp_path / "out")
    foreign = tmp_path / "src"
    foreign.mkdir()
    (foreign / "main.py").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a previous export"):
        export_static(FakeApi(), foreign)
    assert (foreign / "main.py").exists()


def test_export_replaces_a_previous_export(tmp_path: Path) -> None:
    out = tmp_path / "demo-data"
    export_static(FakeApi(), out)
    (out / "runs" / "stale.json").write_text("{}", encoding="utf-8")
    export_static(FakeApi(), out)
    assert not (out / "runs" / "stale.json").exists()
