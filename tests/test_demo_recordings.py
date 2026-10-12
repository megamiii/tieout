from pathlib import Path

import pytest

from tieout.demo_recordings import copy_files, dependencies
from tieout.tracing.tracer import annotate, span, tracing


def test_dependencies_come_from_llm_and_tool_spans() -> None:
    with tracing("agent", "q") as tr, span("agent", "run"):
        with span("llm", "m"):
            annotate(request_hash="aa11")
        with span("tool", "search_filings"):
            annotate(recording_hash="bb22", cached=True)
        with span("tool", "calculate"):  # pure tool: nothing recorded
            pass
    live = dependencies(tr.spans)
    stored = dependencies([{"type": s.type, "meta": s.meta} for s in tr.spans])
    assert live.llm == stored.llm == {"aa11"}
    assert live.tools == stored.tools == {"bb22"}


def test_copy_files_copies_once_and_fails_on_missing(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "ab").mkdir(parents=True)
    (src / "ab" / "abc.json").write_text("{}", encoding="utf-8")
    assert copy_files(["abc"], src, dst) == 1
    assert (dst / "ab" / "abc.json").read_text(encoding="utf-8") == "{}"
    assert copy_files(["abc"], src, dst) == 0  # already there
    with pytest.raises(FileNotFoundError, match="was the run recorded"):
        copy_files(["cd99"], src, dst)
