import json
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

from tieout.agent.finance_tools import CalculateTool
from tieout.agent.tools import Tool, ToolInputError
from tieout.financebench.selection import CompanyConfig, FilingConfig
from tieout.mcp_server.server import build_server


def company(ticker: str, years: list[int]) -> CompanyConfig:
    return CompanyConfig(
        name=f"{ticker} Inc.",
        ticker=ticker,
        cik=1,
        sector="Industrials",
        question_count=4,
        filings=[
            FilingConfig(
                doc_name=f"{ticker}_{y}_10K",
                fiscal_year=y,
                accession=f"0000-{y}",
                form="10-K",
                period_end=f"{y}-12-31",
                filing_date=f"{y + 1}-02-01",
                edgar_url="",
                pdf_url="",
                question_count=1,
            )
            for y in years
        ],
    )


class LookupArgs(BaseModel):
    ticker: str


class LookupTool(Tool):
    """Stands in for a database tool: valid JSON, but an unsupported ticker."""

    name = "lookup"
    description = "Look up a company."
    args_model = LookupArgs

    def run(self, args: LookupArgs) -> dict[str, Any]:
        if args.ticker != "MMM":
            raise ToolInputError(f"unknown ticker {args.ticker!r}; supported: ['MMM']")
        return {"ticker": "MMM", "name": "3M"}


TOOLS: list[Tool] = [LookupTool(), CalculateTool()]
COMPANIES = [company("MMM", [2022, 2018]), company("AMD", [2022])]


def with_client(fn: Any) -> Any:
    async def main() -> Any:
        async with Client(build_server(TOOLS, COMPANIES)) as client:
            return await fn(client)

    return anyio.run(main)


def test_lists_the_agent_tools_with_the_agent_schemas() -> None:
    result = with_client(lambda c: c.list_tools())
    listed = {t.name: t for t in result.tools}
    assert sorted(listed) == ["calculate", "lookup"]
    for tool in TOOLS:
        # One definition: MCP clients see exactly the schema the agent's LLM sees.
        assert listed[tool.name].input_schema == tool.schema()["function"]["parameters"]
        assert listed[tool.name].description == tool.description


def test_instructions_list_the_indexed_filings() -> None:
    async def instructions(c: Client) -> str | None:
        return c.instructions

    text = with_client(instructions)
    assert "- AMD Inc. (AMD): FY2022" in text
    assert "- MMM Inc. (MMM): FY2018, FY2022" in text  # sorted by year


def test_calculate_returns_text_and_structured_content() -> None:
    result = with_client(lambda c: c.call_tool("calculate", {"expression": "1577 - 1373"}))
    assert result.is_error is False
    payload = json.loads(result.content[0].text)
    assert payload == result.structured_content
    assert payload["result"] == "204"


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("lookup", {"ticker": "XYZ"}, "unknown ticker 'XYZ'"),
        ("lookup", {}, "invalid arguments: ticker: Field required"),
        ("calculate", {"expression": "__import__('os')"}, "'__import__' is not allowed"),
    ],
)
def test_bad_calls_are_tool_errors_the_model_can_read(
    name: str, args: dict[str, Any], message: str
) -> None:
    result = with_client(lambda c: c.call_tool(name, args))
    assert result.is_error is True
    assert message in json.loads(result.content[0].text)["error"]


def test_unknown_tool_is_a_protocol_error() -> None:
    async def call(c: Client) -> MCPError | None:
        try:
            await c.call_tool("submit_answer", {})  # agent-only, not exposed
        except MCPError as err:
            return err
        return None

    err = with_client(call)
    assert err is not None and "unknown tool 'submit_answer'" in str(err)
