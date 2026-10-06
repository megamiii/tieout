"""Data pipeline steps: select companies, download sources, load the database.

Each step is idempotent: downloads are cached, and loading replaces a
company's rows, so any step can be re-run safely.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path

import psycopg
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from tieout.config import CONFIG_DIR, Settings
from tieout.db.models import Company, Filing, XbrlFactRow
from tieout.db.session import get_engine, get_sessionmaker, init_db
from tieout.edgar.client import COMPANYFACTS_URL, EdgarClient
from tieout.edgar.xbrl import XbrlFact, iter_facts
from tieout.financebench.dataset import load_documents, load_questions
from tieout.financebench.selection import (
    CompanyConfig,
    read_companies,
    select_companies,
    write_companies,
)
from tieout.net.cached_client import CachedClient, make_client

COMPANIES_PATH = CONFIG_DIR / "companies.json"
FACT_COLUMNS = tuple(f.name for f in fields(XbrlFact))


def client_from(settings: Settings) -> CachedClient:
    return make_client(settings.cache_dir, settings.sec_user_agent, settings.sec_max_rps)


def run_select(http: CachedClient, path: Path = COMPANIES_PATH) -> list[CompanyConfig]:
    docs = load_documents(http)
    questions = load_questions(http)
    companies = select_companies(questions, docs, EdgarClient(http))
    write_companies(companies, path)
    return companies


@dataclass
class DownloadReport:
    pdfs: dict[str, Path] = field(default_factory=dict)
    companyfacts: dict[int, Path] = field(default_factory=dict)

    def total_bytes(self) -> int:
        paths = [*self.pdfs.values(), *self.companyfacts.values()]
        return sum(p.stat().st_size for p in paths)


def run_download(http: CachedClient, companies: list[CompanyConfig]) -> DownloadReport:
    report = DownloadReport()
    for company in companies:
        report.companyfacts[company.cik] = http.path(COMPANYFACTS_URL.format(cik=company.cik))
        for filing in company.filings:
            report.pdfs[filing.doc_name] = http.path(filing.pdf_url)
    return report


def run_load(http: CachedClient, companies: list[CompanyConfig]) -> dict[str, int]:
    """Load companies, filings and XBRL facts into Postgres. Returns row counts."""
    engine = get_engine()
    init_db(engine)
    edgar = EdgarClient(http)
    counts = {"companies": 0, "filings": 0, "xbrl_facts": 0}
    with get_sessionmaker(engine)() as session, session.begin():
        for company in companies:
            stmt = pg_insert(Company).values(
                cik=company.cik, name=company.name, ticker=company.ticker, sector=company.sector
            )
            session.execute(
                stmt.on_conflict_do_update(
                    index_elements=[Company.cik],
                    set_={
                        "name": stmt.excluded.name,
                        "ticker": stmt.excluded.ticker,
                        "sector": stmt.excluded.sector,
                    },
                )
            )
            counts["companies"] += 1

            # Replace this company's filings and facts wholesale (idempotent reload).
            session.execute(delete(Filing).where(Filing.cik == company.cik))
            session.execute(delete(XbrlFactRow).where(XbrlFactRow.cik == company.cik))
            for f in company.filings:
                session.add(
                    Filing(
                        accession=f.accession,
                        cik=company.cik,
                        doc_name=f.doc_name,
                        fiscal_year=f.fiscal_year,
                        form=f.form,
                        period_end=date.fromisoformat(f.period_end),
                        filing_date=date.fromisoformat(f.filing_date),
                        edgar_url=f.edgar_url,
                        pdf_url=f.pdf_url,
                    )
                )
                counts["filings"] += 1
            session.flush()
            facts = iter_facts(edgar.company_facts(company.cik))
            counts["xbrl_facts"] += _copy_facts(session, facts)
    return counts


def _copy_facts(session: Session, facts: Iterable[XbrlFact]) -> int:
    """Bulk-load facts with Postgres COPY, in the session's transaction.

    COPY streams rows in one command instead of building INSERT statements,
    which is what bulk loads need (~135k facts for 13 companies).
    """
    raw = session.connection().connection.driver_connection
    assert isinstance(raw, psycopg.Connection)
    n = 0
    with raw.cursor().copy(f"COPY xbrl_facts ({', '.join(FACT_COLUMNS)}) FROM STDIN") as copy:
        for fact in facts:
            copy.write_row(tuple(getattr(fact, col) for col in FACT_COLUMNS))
            n += 1
    return n


def load_selected(path: Path = COMPANIES_PATH) -> list[CompanyConfig]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `tieout data select` first")
    return read_companies(path)
