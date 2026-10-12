"""MCP server: the agent's data tools, for any MCP client (AI assistants, IDEs).

The server exposes the agent's own `Tool` objects rather than a second set of
functions: the same JSON schemas are listed, and calls go through the same
validate-then-run path as the agent's tool calls, so an MCP client gets the
same argument checks and the same `{"error": ...}` messages to correct itself
from. `submit_answer` is not exposed: it only means something inside the
agent loop, where it is validated against the run's evidence.

No LLM runs here; the client's own model does the reasoning, so serving
these tools costs nothing.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError

from tieout.agent.finance_tools import CalculateTool, GetFinancialFactsTool, ReadSectionTool
from tieout.agent.tools import SearchFilingsTool, Tool, execute_tool_call
from tieout.financebench.selection import CompanyConfig
from tieout.llm.types import ToolCall

log = logging.getLogger("tieout.mcp")

SERVER_NAME = "tieout"
METHOD_NOT_FOUND = -32601


def coverage(companies: Sequence[CompanyConfig]) -> str:
    """Which filings the tools can answer about, one line per company."""
    lines = []
    for c in sorted(companies, key=lambda c: c.ticker):
        years = ", ".join(f"FY{y}" for y in sorted(f.fiscal_year for f in c.filings))
        lines.append(f"- {c.name} ({c.ticker}): {years}")
    return "\n".join(lines)


def instructions(companies: Sequence[CompanyConfig]) -> str:
    return (
        "Tools over SEC 10-K filings and their XBRL financial data. Only these "
        "filings are indexed:\n"
        f"{coverage(companies)}\n\n"
        "Use get_financial_facts for exact line-item values (XBRL, as originally "
        "reported), search_filings and read_section for the text (MD&A, notes), and "
        "calculate for any arithmetic. Cite chunk_id (text) and fact_id (XBRL) "
        "values from the results so answers can be checked."
    )


def build_server(tools: Sequence[Tool], companies: Sequence[CompanyConfig]) -> Server[Any]:
    by_name = {t.name: t for t in tools}
    listed = [
        types.Tool(
            name=t.name,
            description=t.description,
            input_schema=t.schema()["function"]["parameters"],
        )
        for t in tools
    ]

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name not in by_name:
            # An unknown tool is a protocol error; a bad call to a real tool is a
            # tool result with is_error, which the client's model can act on.
            raise MCPError(METHOD_NOT_FOUND, f"unknown tool {params.name!r}")
        call = ToolCall(id="mcp", name=params.name, arguments=json.dumps(params.arguments or {}))
        # Tools are blocking (database, embedding model): keep the event loop free.
        outcome = await anyio.to_thread.run_sync(execute_tool_call, by_name, call)
        if outcome.error:
            log.info("tool %s rejected: %s", params.name, outcome.error)
        else:
            log.info("tool %s ok in %.0f ms", params.name, outcome.latency_ms)
        return types.CallToolResult(
            content=[types.TextContent(text=outcome.content)],
            structured_content=outcome.output if isinstance(outcome.output, dict) else None,
            is_error=outcome.error is not None,
        )

    return Server(
        SERVER_NAME,
        version="0.1.0",
        instructions=instructions(companies),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def default_tools(companies: list[CompanyConfig]) -> list[Tool]:
    """The live data tools, reading Postgres directly (no tool-output cache)."""
    from tieout.agent.factory import _LazyEmbedder
    from tieout.config import get_settings
    from tieout.db.session import get_sessionmaker

    settings = get_settings()
    sessions = get_sessionmaker()
    embedder = _LazyEmbedder(settings)
    return [
        SearchFilingsTool(sessions, embedder, companies),
        ReadSectionTool(sessions, companies),
        GetFinancialFactsTool(sessions, embedder, companies),
        CalculateTool(),
    ]


def run_stdio(server: Server[Any]) -> None:
    """Serve over stdin/stdout. stdout carries the protocol, so logs go to stderr."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    async def main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(main)
