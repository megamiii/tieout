"""Export the public demo as static JSON, for GitHub Pages.

The demo only replays recorded runs, so everything it can show is fixed data.
This module calls the real demo API endpoints (in-process, in demo mode) and
writes each response to a file; the frontend's static build reads those files
instead of calling a server. The static site therefore shows exactly what the
demo server would, with no server, database or API key.

Layout (relative to the output directory):
    config.json, companies.json, eval.json
    asks.json                       question text -> run id (for "Ask")
    runs/queue-<queue>.json         review queue lists
    runs/<run id>.json              run details
    workflows/list-<kind>.json      workflow lists
    workflows/<run id>.json         workflow details
    traces/<trace id>.json          span trees
"""

import json
import shutil
from pathlib import Path
from typing import Any, Protocol

QUEUES = ("pending", "reviewed", "all")
WORKFLOW_KINDS = ("tieout", "analytical_review")


class _Response(Protocol):
    def json(self) -> Any: ...

    def raise_for_status(self) -> Any: ...


class ApiClient(Protocol):
    def get(self, url: str) -> _Response: ...

    def post(self, url: str, *, json: Any) -> _Response: ...


def _clear(out_dir: Path) -> None:
    """Remove a previous export, but never a directory that isn't one."""
    if out_dir.exists():
        if any(out_dir.iterdir()) and not (out_dir / "config.json").exists():
            raise RuntimeError(f"{out_dir} is not empty and is not a previous export")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def export_static(client: ApiClient, out_dir: Path) -> dict[str, int]:
    _clear(out_dir)
    counts: dict[str, int] = {}

    def get(path: str) -> Any:
        response = client.get(path)
        response.raise_for_status()
        return response.json()

    def write(rel: str, data: Any) -> None:
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), "utf-8")
        kind = rel.split("/")[0] if "/" in rel else "other"
        counts[kind] = counts.get(kind, 0) + 1

    config = get("/api/config")
    if not config.get("demo_mode"):
        raise RuntimeError("the API is not in demo mode; the export would not be replay-only")
    write("config.json", {**config, "static": True})
    write("companies.json", get("/api/companies"))
    write("eval.json", get("/api/eval"))

    run_ids: set[str] = set()
    trace_ids: set[str] = set()
    asks: dict[str, str] = {}
    for q in config["demo_questions"]:
        response = client.post("/api/ask", json={"question": q["question"]})
        response.raise_for_status()
        asks[q["question"]] = response.json()["id"]
    write("asks.json", asks)
    run_ids.update(asks.values())

    for queue in QUEUES:
        rows = get(f"/api/runs?queue={queue}")
        write(f"runs/queue-{queue}.json", rows)
        run_ids.update(r["id"] for r in rows)
    for run_id in sorted(run_ids):
        detail = get(f"/api/runs/{run_id}")
        write(f"runs/{run_id}.json", detail)
        if detail.get("trace_id"):
            trace_ids.add(detail["trace_id"])

    for kind in WORKFLOW_KINDS:
        rows = get(f"/api/workflows?kind={kind}")
        write(f"workflows/list-{kind}.json", rows)
        for row in rows:
            detail = get(f"/api/workflows/{row['id']}")
            write(f"workflows/{row['id']}.json", detail)
            if detail.get("trace_id"):
                trace_ids.add(detail["trace_id"])

    for trace_id in sorted(trace_ids):
        write(f"traces/{trace_id}.json", get(f"/api/traces/{trace_id}"))
    return counts
