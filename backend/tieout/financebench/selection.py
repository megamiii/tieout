"""Choose which companies tieout supports, by a reproducible rule.

Rule: every company with at least MIN_10K_QUESTIONS FinanceBench questions
about a 10-K. The result (with SEC identifiers and the matching EDGAR
filing for each document) is written to config/companies.json.
"""

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from tieout.edgar.client import EdgarClient
from tieout.edgar.filings import match_annual_filing
from tieout.financebench.dataset import Document, Question

MIN_10K_QUESTIONS = 4

# FinanceBench company name -> stock ticker (used to look up the SEC CIK).
TICKERS = {
    "3M": "MMM",
    "AES Corporation": "AES",
    "AMD": "AMD",
    "Adobe": "ADBE",
    "Amazon": "AMZN",
    "Amcor": "AMCR",
    "American Express": "AXP",
    "American Water Works": "AWK",
    "Best Buy": "BBY",
    "Boeing": "BA",
    "CVS Health": "CVS",
    "Coca-Cola": "KO",
    "Corning": "GLW",
    "Costco": "COST",
    "General Mills": "GIS",
    "JPMorgan": "JPM",
    "Johnson & Johnson": "JNJ",
    "Kraft Heinz": "KHC",
    "Lockheed Martin": "LMT",
    "MGM Resorts": "MGM",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "Paypal": "PYPL",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Ulta Beauty": "ULTA",
    "Verizon": "VZ",
    "Walmart": "WMT",
}


class FilingConfig(BaseModel):
    doc_name: str  # FinanceBench document name, e.g. "AMD_2022_10K"
    fiscal_year: int
    accession: str
    form: str
    period_end: str  # ISO date
    filing_date: str  # ISO date
    edgar_url: str
    pdf_url: str
    question_count: int


class CompanyConfig(BaseModel):
    name: str
    ticker: str
    cik: int
    sector: str
    question_count: int
    filings: list[FilingConfig]


def tenk_questions(questions: list[Question], docs: dict[str, Document]) -> list[Question]:
    return [q for q in questions if docs[q.doc_name].doc_type == "10k"]


def select_companies(
    questions: list[Question],
    docs: dict[str, Document],
    edgar: EdgarClient,
    min_questions: int = MIN_10K_QUESTIONS,
) -> list[CompanyConfig]:
    tenk = tenk_questions(questions, docs)
    per_company = Counter(q.company for q in tenk)
    per_doc = Counter(q.doc_name for q in tenk)
    chosen = [c for c, n in per_company.most_common() if n >= min_questions]

    cik_by_ticker = edgar.ticker_to_cik()
    companies: list[CompanyConfig] = []
    for name in chosen:
        ticker = TICKERS[name]
        cik = cik_by_ticker[ticker]
        refs = edgar.filings(cik)
        filings: list[FilingConfig] = []
        for doc_name in sorted(per_doc):
            doc = docs[doc_name]
            if doc.company != name:
                continue
            ref = match_annual_filing(refs, doc.doc_period, link_hint=doc.doc_link)
            assert ref.report_date is not None
            filings.append(
                FilingConfig(
                    doc_name=doc_name,
                    fiscal_year=doc.doc_period,
                    accession=ref.accession,
                    form=ref.form,
                    period_end=ref.report_date.isoformat(),
                    filing_date=ref.filing_date.isoformat(),
                    edgar_url=ref.url(cik),
                    pdf_url=doc.pdf_url,
                    question_count=per_doc[doc_name],
                )
            )
        sector = next(d.gics_sector for d in docs.values() if d.company == name)
        companies.append(
            CompanyConfig(
                name=name,
                ticker=ticker,
                cik=cik,
                sector=sector,
                question_count=per_company[name],
                filings=filings,
            )
        )
    return companies


def write_companies(companies: list[CompanyConfig], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.model_dump() for c in companies]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_companies(path: Path) -> list[CompanyConfig]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [CompanyConfig.model_validate(r) for r in rows]
