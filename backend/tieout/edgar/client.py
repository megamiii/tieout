"""SEC EDGAR endpoints we use: ticker map, submissions and XBRL company facts.

Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
All requests go through CachedClient, so each URL is fetched once.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

from tieout.net.cached_client import CachedClient

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{document}"

ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT")


@dataclass(frozen=True)
class FilingRef:
    accession: str  # e.g. "0000066740-23-000014"
    form: str
    filing_date: date
    report_date: date | None  # period end of the report
    primary_document: str

    def url(self, cik: int) -> str:
        return ARCHIVE_URL.format(
            cik=cik, acc_nodash=self.accession.replace("-", ""), document=self.primary_document
        )


class EdgarClient:
    def __init__(self, http: CachedClient) -> None:
        self.http = http

    def ticker_to_cik(self) -> dict[str, int]:
        data: dict[str, dict[str, Any]] = self.http.get_json(TICKERS_URL)
        return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}

    def company_facts(self, cik: int) -> dict[str, Any]:
        facts: dict[str, Any] = self.http.get_json(COMPANYFACTS_URL.format(cik=cik))
        return facts

    def filings(self, cik: int, forms: tuple[str, ...] = ANNUAL_FORMS) -> list[FilingRef]:
        """All filings of the given forms, including those in older pages.

        The submissions JSON keeps the ~1000 most recent filings inline
        ("recent") and lists older ones in extra files; companies with many
        insider-trading filings push their older 10-Ks into those pages.
        """
        main = self.http.get_json(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        tables = [main["filings"]["recent"]]
        for page in main["filings"].get("files", []):
            tables.append(self.http.get_json(SUBMISSIONS_URL.format(name=page["name"])))
        refs: dict[str, FilingRef] = {}
        for table in tables:
            for i, form in enumerate(table["form"]):
                if form not in forms:
                    continue
                report = table["reportDate"][i]
                ref = FilingRef(
                    accession=table["accessionNumber"][i],
                    form=form,
                    filing_date=date.fromisoformat(table["filingDate"][i]),
                    report_date=date.fromisoformat(report) if report else None,
                    primary_document=table["primaryDocument"][i],
                )
                refs[ref.accession] = ref
        return sorted(refs.values(), key=lambda r: r.filing_date)
