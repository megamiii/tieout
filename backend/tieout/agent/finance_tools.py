"""read_section, get_financial_facts, calculate and submit_answer."""

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from tieout.agent.calculator import CalcError, evaluate, format_decimal
from tieout.agent.tools import Tool, ToolInputError, chunk_evidence
from tieout.agent.validation import Evidence, SubmitAnswerArgs
from tieout.db.models import ChunkRow, SectionRow
from tieout.edgar.facts_query import annual_values, find_concepts, fiscal_year_ends
from tieout.financebench.selection import CompanyConfig, FilingConfig
from tieout.retrieval.embeddings import Embedder

PAGE_CHARS = 6000  # one page of chunks; Item 8 alone can be 70 pages
MAX_CONCEPTS = 3


class _Companies:
    def __init__(self, companies: list[CompanyConfig]) -> None:
        self.by_ticker = {c.ticker: c for c in companies}

    def company(self, ticker: str) -> CompanyConfig:
        company = self.by_ticker.get(ticker.upper())
        if company is None:
            raise ToolInputError(f"unknown ticker {ticker!r}; supported: {sorted(self.by_ticker)}")
        return company

    def filing(self, ticker: str, fiscal_year: int) -> FilingConfig:
        company = self.company(ticker)
        for f in company.filings:
            if f.fiscal_year == fiscal_year:
                return f
        years = [f.fiscal_year for f in company.filings]
        raise ToolInputError(f"no {company.ticker} 10-K for FY{fiscal_year}; available: {years}")


# --- read_section ---------------------------------------------------------------


class ReadSectionArgs(BaseModel):
    ticker: str
    fiscal_year: int
    item: str = Field(description="10-K item, e.g. '7' (MD&A), '8' (financial statements), '1A'.")
    page: int | None = Field(
        default=None, description="Page to read within the item; default: its first page."
    )


class ReadSectionTool(Tool):
    name = "read_section"
    description = (
        "Read one page of a 10-K item in full (e.g. the whole cash flow statement around a "
        "search hit). Returns the page's passages with chunk_ids, and the item's page range."
    )
    args_model = ReadSectionArgs

    def __init__(
        self, session_factory: Callable[[], Session], companies: list[CompanyConfig]
    ) -> None:
        self.session_factory = session_factory
        self.companies = _Companies(companies)

    def run(self, args: ReadSectionArgs) -> dict[str, Any]:
        filing = self.companies.filing(args.ticker, args.fiscal_year)
        item = args.item.upper().removeprefix("ITEM").strip()
        with self.session_factory() as session:
            section = session.scalar(
                select(SectionRow).where(
                    SectionRow.accession == filing.accession, SectionRow.item == item
                )
            )
            if section is None:
                items = list(
                    session.scalars(
                        select(SectionRow.item).where(SectionRow.accession == filing.accession)
                    )
                )
                raise ToolInputError(f"no Item {item} in this filing; items: {items}")
            page = args.page or section.page_start
            if not section.page_start <= page <= section.page_end:
                raise ToolInputError(
                    f"Item {item} spans pages {section.page_start}-{section.page_end}"
                )
            rows = session.scalars(
                select(ChunkRow)
                .where(
                    ChunkRow.accession == filing.accession,
                    ChunkRow.section == item,
                    ChunkRow.page == page,
                )
                .order_by(ChunkRow.ordinal)
            )
            chunks: list[dict[str, Any]] = []
            used = 0
            for r in rows:
                text = " ".join(r.text.split())
                if used + len(text) > PAGE_CHARS and chunks:
                    break
                used += len(text)
                chunks.append({"chunk_id": r.chunk_id, "page": page, "text": text})
        return {
            "ticker": args.ticker.upper(),
            "fiscal_year": args.fiscal_year,
            "item": item,
            "title": section.title,
            "page": page,
            "item_pages": [section.page_start, section.page_end],
            "chunks": chunks,
        }

    def evidence(self, output: Any) -> list[Evidence]:
        return chunk_evidence(output.get("chunks", []))


# --- get_financial_facts --------------------------------------------------------


class GetFinancialFactsArgs(BaseModel):
    ticker: str
    query: str | None = Field(
        default=None,
        description="Line item in plain words, e.g. 'capital expenditure', 'total revenue'.",
    )
    concept: str | None = Field(
        default=None, description="Exact US-GAAP concept name, if known (overrides query)."
    )
    fiscal_years: list[int] | None = Field(
        default=None, max_length=5, description="Fiscal years; default: the latest 3."
    )


class GetFinancialFactsTool(Tool):
    name = "get_financial_facts"
    description = (
        "Structured financial data (XBRL) that the company reported to the SEC in its 10-Ks. "
        "Returns annual values as originally reported, each with a fact_id to cite, for the "
        "best-matching concepts. Use it for any line-item figure."
    )
    args_model = GetFinancialFactsArgs

    def __init__(
        self,
        session_factory: Callable[[], Session],
        embedder: Embedder,
        companies: list[CompanyConfig],
    ) -> None:
        self.session_factory = session_factory
        self.embedder = embedder
        self.companies = _Companies(companies)

    def run(self, args: GetFinancialFactsArgs) -> dict[str, Any]:
        company = self.companies.company(args.ticker)
        if not args.query and not args.concept:
            raise ToolInputError("give a query (line item in words) or an exact concept")
        with self.session_factory() as session:
            if args.concept:
                concepts = [args.concept]
            else:
                assert args.query is not None
                concepts = find_concepts(
                    session, self.embedder, company.cik, args.query, k=MAX_CONCEPTS
                )
            year_ends = fiscal_year_ends(session, company.cik)
            results = []
            for concept in concepts:
                for cv in annual_values(
                    session, company.cik, concept, args.fiscal_years, year_ends
                ):
                    results.append(
                        {
                            "concept": cv.concept,
                            "label": cv.label,
                            "unit": cv.unit,
                            "values": [
                                {
                                    "fiscal_year": v.fiscal_year,
                                    "period": (
                                        f"{v.period_start}..{v.period_end}"
                                        if v.period_start
                                        else f"as of {v.period_end}"
                                    ),
                                    "value": format_decimal(v.value),
                                    "fact_id": v.fact_id,
                                    "reported_in_10k_filed": v.filed.isoformat(),
                                    **({"later_restated_to": v.restated} if v.restated else {}),
                                }
                                for v in cv.values
                            ],
                        }
                    )
        if not results:
            return {"ticker": company.ticker, "concepts": [], "note": "no annual values found"}
        return {"ticker": company.ticker, "concepts": results}

    def evidence(self, output: Any) -> list[Evidence]:
        out = []
        for c in output.get("concepts", []):
            for v in c["values"]:
                out.append(
                    Evidence(
                        source_id=v["fact_id"],
                        kind="fact",
                        text=f"{c['concept']} {c['label']} {c['unit']} "
                        f"FY{v['fiscal_year']}: {v['value']}",
                        values=[Decimal(v["value"])],
                        meta={
                            "ticker": output["ticker"],
                            "concept": c["concept"],
                            "fiscal_year": v["fiscal_year"],
                            "restated": v.get("later_restated_to", []),
                        },
                    )
                )
        return out


# --- calculate ------------------------------------------------------------------


class CalculateArgs(BaseModel):
    expression: str = Field(
        description="Arithmetic only, e.g. '(1577 - 1373) / 1373 * 100'. "
        "Functions: abs, round, min, max."
    )


class CalculateTool(Tool):
    name = "calculate"
    description = (
        "Evaluate an arithmetic expression exactly. Use it for every computed number "
        "(differences, ratios, growth rates, sums); never do arithmetic yourself."
    )
    args_model = CalculateArgs

    def run(self, args: CalculateArgs) -> dict[str, str]:
        try:
            result = evaluate(args.expression)
        except CalcError as err:
            raise ToolInputError(str(err)) from err
        return {"expression": args.expression, "result": format_decimal(result)}

    def evidence(self, output: Any) -> list[Evidence]:
        return [
            Evidence(
                source_id=f"calc:{output['expression']}",
                kind="calc",
                text=output["expression"],
                values=[Decimal(output["result"])],
            )
        ]


# --- submit_answer --------------------------------------------------------------


class SubmitAnswerTool(Tool):
    """Ends the run. The loop validates the arguments against the run's evidence."""

    name = "submit_answer"
    description = (
        "Submit the final answer. Every citation needs a source_id returned by a tool "
        "and a quote that appears in it. Numbers in the answer must come from cited "
        "sources or calculate results. Set needs_review when evidence is missing or "
        "conflicting."
    )
    args_model = SubmitAnswerArgs

    def run(self, args: SubmitAnswerArgs) -> dict[str, str]:
        return {"status": "received"}
