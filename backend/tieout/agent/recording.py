"""Record and replay tool outputs, like the LLM response cache.

A tool call is keyed by a hash of (tool name, canonical JSON arguments). With
both caches recorded, an agent run replays without an API key *and* without
the database or embedding model, which is what CI and the public demo need.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from tieout.agent.tools import Tool, ToolInputError
from tieout.agent.validation import Evidence
from tieout.tracing.tracer import annotate


class ToolReplayMiss(Exception):
    """Offline mode: this tool call was not recorded."""


def tool_call_hash(name: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps({"tool": name, "arguments": arguments}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RecordedTool(Tool):
    def __init__(self, inner: Tool, cache_dir: Path, *, offline: bool = False) -> None:
        self.inner = inner
        self.name = inner.name
        self.description = inner.description
        self.args_model = inner.args_model
        self.cache_dir = cache_dir
        self.offline = offline

    def run(self, args: Any) -> Any:
        arguments = args.model_dump(mode="json")
        h = tool_call_hash(self.name, arguments)
        path = self.cache_dir / h[:2] / f"{h}.json"
        if path.exists():
            annotate(cached=True)
            record = json.loads(path.read_text(encoding="utf-8"))
            if "error" in record:
                raise ToolInputError(record["error"])
            return record["output"]
        if self.offline:
            raise ToolReplayMiss(f"{self.name}({arguments}) was not recorded")
        try:
            output = self.inner.run(args)
        except ToolInputError as err:
            self._write(path, {"tool": self.name, "arguments": arguments, "error": str(err)})
            raise
        self._write(path, {"tool": self.name, "arguments": arguments, "output": output})
        return output

    def evidence(self, output: Any) -> list[Evidence]:
        return self.inner.evidence(output)

    @staticmethod
    def _write(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".part-")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
