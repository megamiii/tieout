"""PDF -> list of page texts (PyMuPDF)."""

import re
from datetime import date, datetime
from pathlib import Path

import pymupdf


def extract_pages(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        return [page.get_text() for page in doc]


FY_END_RE = re.compile(
    r"fiscal\s+year\s+ended:?\s*([A-Z][a-z]+)\s*(\d{1,2})\s*,\s*(\d{4})", re.IGNORECASE
)


def cover_fiscal_year_end(pages: list[str], max_pages: int = 3) -> date | None:
    """The 'For the fiscal year ended <date>' on the cover page, if found."""
    text = re.sub(r"\s+", " ", " ".join(pages[:max_pages]))
    m = FY_END_RE.search(text)
    if not m:
        return None
    try:
        return datetime.strptime(" ".join(m.groups()), "%B %d %Y").date()
    except ValueError:
        return None
