"""Tools the agent can call, with Pydantic-validated arguments.

Each tool declares an argument model; its JSON schema becomes the tool's
`parameters` for the LLM, and the same model validates what the LLM sends
back. Validation errors are returned to the model as tool output so it can
correct itself, instead of crashing the run.
"""

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from tieout.agent.validation import Evidence, numbers_in
from tieout.financebench.selection import CompanyConfig
from tieout.llm.types import ToolCall
from tieout.retrieval.embeddings import Embedder
from tieout.retrieval.search import Mode, SearchFilters, search
from tieout.tracing.tracer import span

SNIPPET_CHARS = 1000  # ~250 tokens per hit: free-tier tokens/min are tight


def _strip_titles(schema: Any) -> Any:
    """Drop Pydantic's auto-generated 'title' keys: they cost tokens and add nothing.

    Keys of a "properties" mapping are field names, so a field called "title"
    is kept.
    """
    if isinstance(schema, dict):
        return {
            k: (
                {name: _strip_titles(sub) for name, sub in v.items()}
                if k == "properties" and isinstance(v, dict)
                else _strip_titles(v)
            )
            for k, v in schema.items()
            if k != "title"
        }
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


class Tool(ABC):
    name: str
    description: str
    args_model: type[BaseModel]

    @abstractmethod
    def run(self, args: Any) -> Any:
        """Execute with validated args; return something JSON-serializable."""

    def evidence(self, output: Any) -> list[Evidence]:
        """Sources (or calculation results) this output showed the model."""
        return []

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strip_titles(self.args_model.model_json_schema()),
            },
        }


class ToolInputError(Exception):
    """Raised by a tool for bad-but-valid-JSON input; the message goes to the model."""


@dataclass
class ToolOutcome:
    call_id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any] | None
    output: Any
    error: str | None
    latency_ms: float
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def content(self) -> str:
        """What the model sees as the tool message."""
        if self.error:
            return json.dumps({"error": self.error})
        return json.dumps(self.output, ensure_ascii=False)


def execute_tool_call(tools: dict[str, Tool], call: ToolCall) -> ToolOutcome:
    """Run one tool call, recorded as a `tool` span when tracing is on."""
    with span("tool", call.name, input={"arguments": call.arguments}) as s:
        result = _execute_tool_call(tools, call)
        s.output = result.output
        s.error = result.error
        s.meta.update({"sources": [e.source_id for e in result.evidence]})
    return result


def _execute_tool_call(tools: dict[str, Tool], call: ToolCall) -> ToolOutcome:
    start = time.perf_counter()

    def outcome(args: dict[str, Any] | None, output: Any, error: str | None) -> ToolOutcome:
        return ToolOutcome(
            call_id=call.id,
            name=call.name,
            raw_arguments=call.arguments,
            arguments=args,
            output=output,
            error=error,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    tool = tools.get(call.name)
    if tool is None:
        return outcome(None, None, f"unknown tool {call.name!r}; available: {sorted(tools)}")
    try:
        parsed = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as err:
        return outcome(None, None, f"arguments are not valid JSON: {err}")
    try:
        args = tool.args_model.model_validate(parsed)
    except ValidationError as err:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors())
        return outcome(parsed, None, f"invalid arguments: {problems}")
    try:
        output = tool.run(args)
    except ToolInputError as err:
        return outcome(args.model_dump(), None, str(err))
    result = outcome(args.model_dump(), output, None)
    result.evidence = tool.evidence(output)
    return result


def chunk_evidence(chunks: list[dict[str, Any]]) -> list[Evidence]:
    return [
        Evidence(
            source_id=c["chunk_id"],
            kind="chunk",
            text=c["text"],
            values=numbers_in(c["text"]),
            meta={k: c[k] for k in ("ticker", "fiscal_year", "section", "page") if k in c},
        )
        for c in chunks
    ]


class SearchFilingsArgs(BaseModel):
    query: str = Field(min_length=2, description="What to look for, in the filing's own words.")
    ticker: str | None = Field(default=None, description="Company ticker, e.g. 'MMM'.")
    fiscal_year: int | None = Field(default=None, description="Fiscal year of the 10-K.")
    section: str | None = Field(
        default=None,
        description="10-K item: '1' business, '1A' risk factors, '7' MD&A, "
        "'7A' market risk, '8' financial statements.",
    )
    k: int = Field(default=5, ge=1, le=10, description="Number of passages to return.")


class SearchFilingsTool(Tool):
    name = "search_filings"
    description = (
        "Search the text of supported 10-K filings. Returns passages with a chunk_id, "
        "company, fiscal year, 10-K item and page. Filter by ticker and fiscal_year "
        "whenever the question names them."
    )
    args_model = SearchFilingsArgs

    def __init__(
        self,
        session_factory: Callable[[], Session],
        embedder: Embedder,
        companies: list[CompanyConfig],
        mode: Mode = "hybrid",
    ) -> None:
        self.session_factory = session_factory
        self.embedder = embedder
        self.mode = mode
        self.by_ticker = {c.ticker: c for c in companies}
        self.by_accession = {f.accession: c for c in companies for f in c.filings}

    def run(self, args: SearchFilingsArgs) -> dict[str, Any]:
        cik = None
        if args.ticker:
            company = self.by_ticker.get(args.ticker.upper())
            if company is None:
                raise ToolInputError(
                    f"unknown ticker {args.ticker!r}; supported: {sorted(self.by_ticker)}"
                )
            cik = company.cik
        filters = SearchFilters(cik=cik, fiscal_year=args.fiscal_year, section=args.section)
        with self.session_factory() as session:
            hits = search(session, self.embedder, args.query, filters, k=args.k, mode=self.mode)
        results = []
        for h in hits:
            company = self.by_accession[h.accession]
            text = " ".join(h.text.split())
            results.append(
                {
                    "chunk_id": h.chunk_id,
                    "ticker": company.ticker,
                    "fiscal_year": h.fiscal_year,
                    "section": f"Item {h.section}" if h.section[0].isdigit() else h.section,
                    "page": h.page,
                    "text": text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else ""),
                }
            )
        if not results:
            return {"results": [], "note": "no passages matched; try other words or filters"}
        return {"results": results}

    def evidence(self, output: Any) -> list[Evidence]:
        return chunk_evidence(output.get("results", []))
