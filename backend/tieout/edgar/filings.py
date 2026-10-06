"""Match a fiscal year label (e.g. FinanceBench's "AMD_2022_10K") to an EDGAR 10-K."""

import re
from datetime import date

from tieout.edgar.client import FilingRef

ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}")


def fiscal_year_label(period_end: date) -> int:
    """The fiscal year a period end date belongs to.

    Companies name a fiscal year after the calendar year it ends in (Nike's
    FY2023 ends 2023-05-31, Best Buy's FY2023 ends 2023-01-28). 52/53-week
    years that end in the first days of January belong to the prior year
    (a year ending 2021-01-02 is FY2020).
    """
    if period_end.month == 1 and period_end.day <= 7:
        return period_end.year - 1
    return period_end.year


def accession_in_link(link: str) -> str | None:
    match = ACCESSION_RE.search(link)
    return match.group(0) if match else None


class FilingMatchError(Exception):
    pass


def match_annual_filing(
    filings: list[FilingRef], fiscal_year: int, link_hint: str | None = None
) -> FilingRef:
    """Pick the original 10-K whose period ends in `fiscal_year`.

    If `link_hint` (a document URL) contains an accession number, it must
    agree with the date-based match; disagreement raises instead of guessing.
    """
    candidates = [
        f for f in filings if f.report_date and fiscal_year_label(f.report_date) == fiscal_year
    ]
    originals = [f for f in candidates if f.form == "10-K"]
    pool = originals or candidates
    if not pool:
        raise FilingMatchError(f"no annual filing for fiscal year {fiscal_year}")
    # If several originals exist (rare), the earliest filed is the original report.
    chosen = min(pool, key=lambda f: f.filing_date)

    hinted = accession_in_link(link_hint) if link_hint else None
    if hinted and hinted != chosen.accession:
        raise FilingMatchError(
            f"FY{fiscal_year}: date match {chosen.accession} != link accession {hinted}"
        )
    return chosen
