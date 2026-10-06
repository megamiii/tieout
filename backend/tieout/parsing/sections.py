"""Split a 10-K's page texts into its standard items (Item 1A, Item 7, ...).

Pure functions over a list of page strings, so they are easy to unit test.

Three traps this handles:
- The table of contents lists every heading. TOC pages (many headings, early
  in the document) are skipped.
- Running text references items ("see Item 7"). A heading must start a line
  and its line (or the next line) must contain the item's title keyword.
- PDF extraction splits words ("Busines s"), so keywords are compared with
  everything but letters removed.
"""

import re
from dataclasses import dataclass

# Item code -> (canonical title, keywords; any one must appear in the heading)
ITEMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "1": ("Business", ("business",)),
    "1A": ("Risk Factors", ("riskfactors",)),
    "1B": ("Unresolved Staff Comments", ("unresolved",)),
    "1C": ("Cybersecurity", ("cybersecurity",)),
    "2": ("Properties", ("properties",)),
    "3": ("Legal Proceedings", ("legalproceedings",)),
    "4": ("Mine Safety Disclosures", ("minesafety", "submissionofmatters")),
    "5": ("Market for Registrant's Common Equity", ("marketfor", "marketforthe")),
    "6": ("Selected Financial Data", ("selectedfinancial", "reserved", "selected")),
    "7": ("Management's Discussion and Analysis", ("discussionandanalysis",)),
    "7A": ("Quantitative and Qualitative Disclosures About Market Risk", ("quantitative",)),
    "8": ("Financial Statements and Supplementary Data", ("financialstatements",)),
    "9": ("Changes in and Disagreements with Accountants", ("changesinanddisagreements",)),
    "9A": ("Controls and Procedures", ("controlsandprocedures",)),
    "9B": ("Other Information", ("otherinformation",)),
    "9C": ("Disclosure Regarding Foreign Jurisdictions", ("foreignjurisdictions",)),
    "10": ("Directors, Executive Officers and Corporate Governance", ("directors",)),
    "11": ("Executive Compensation", ("executivecompensation",)),
    "12": ("Security Ownership", ("securityownership",)),
    "13": ("Certain Relationships and Related Transactions", ("certainrelationships",)),
    "14": ("Principal Accountant Fees and Services", ("principalaccount",)),
    "15": ("Exhibits and Financial Statement Schedules", ("exhibits",)),
    "16": ("Form 10-K Summary", ("summary",)),
}
ORDER = {code: i for i, code in enumerate(ITEMS)}

# Text before the first item (cover page, TOC, forward-looking statements).
FRONT = "front"

HEADING_RE = re.compile(r"^\s*item\s*(\d{1,2}\s*[a-c]?)\b(.*)$", re.IGNORECASE)
TOC_MIN_HEADINGS = 5
TOC_MAX_POSITION = 0.15  # TOC pages are in the first 15% of the document
MAX_HEADING_LEN = 200


@dataclass(frozen=True)
class Heading:
    item: str
    page: int  # 0-based page index
    line: int  # line index within the page


@dataclass(frozen=True)
class Section:
    item: str  # "1A", "7", ... or FRONT
    title: str
    page_start: int  # 1-based, inclusive
    page_end: int  # 1-based, inclusive
    # (1-based page number, text of this section on that page)
    pages: tuple[tuple[int, str], ...]

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.pages)


def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def clean_page(text: str) -> str:
    """Normalize extraction artifacts that would hurt matching and search."""
    text = text.replace("\ufffd", "'").replace("\u2019", "'").replace("\u00a0", " ")
    # Control characters (including NUL, which Postgres text rejects), except newline.
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def _heading_at(lines: list[str], i: int) -> str | None:
    """Item code if lines[i] is a real heading, else None."""
    line = lines[i]
    if len(line) > MAX_HEADING_LEN:
        return None
    m = HEADING_RE.match(line)
    if not m:
        return None
    code = re.sub(r"\s", "", m.group(1)).upper()
    if code not in ITEMS:
        return None
    # "Item 7, Management's Discussion ..., and the Notes" is a cross-reference
    # in running text; real headings use ".", ":", a dash, or nothing.
    if m.group(2).lstrip().startswith(","):
        return None
    # Title keyword on this line or (for "ITEM 7" alone on a line) the next one.
    window = letters(m.group(2))
    if len(window) < 4 and i + 1 < len(lines):
        window += letters(lines[i + 1])
    keywords = ITEMS[code][1]
    if not any(window.startswith(k) or k in window[:60] for k in keywords):
        return None
    return code


def find_headings(pages: list[str]) -> list[Heading]:
    """Body headings in document order, skipping TOC pages and out-of-order hits."""
    candidates: list[list[Heading]] = []
    for p, text in enumerate(pages):
        lines = text.splitlines()
        found = [Heading(code, p, i) for i in range(len(lines)) if (code := _heading_at(lines, i))]
        candidates.append(found)

    n = len(pages)
    accepted: list[Heading] = []
    last = -1
    for p, found in enumerate(candidates):
        is_toc = len(found) >= TOC_MIN_HEADINGS and p < max(1, int(n * TOC_MAX_POSITION))
        if is_toc:
            continue
        for h in found:
            if ORDER[h.item] > last:
                accepted.append(h)
                last = ORDER[h.item]
    return accepted


def split_sections(pages: list[str]) -> list[Section]:
    """Split pages into sections. Every line belongs to exactly one section."""
    pages = [clean_page(p) for p in pages]
    headings = find_headings(pages)
    # Boundaries: (page, line, item); the front matter starts at (0, 0).
    bounds = [(0, 0, FRONT), *[(h.page, h.line, h.item) for h in headings]]
    sections: list[Section] = []
    for k, (page, line, item) in enumerate(bounds):
        end_page, end_line = (
            (bounds[k + 1][0], bounds[k + 1][1]) if k + 1 < len(bounds) else (len(pages) - 1, None)
        )
        parts: list[tuple[int, str]] = []
        for p in range(page, end_page + 1):
            lines = pages[p].splitlines()
            start = line if p == page else 0
            stop = end_line if (p == end_page and end_line is not None) else len(lines)
            chunk = "\n".join(lines[start:stop]).strip()
            if chunk:
                parts.append((p + 1, chunk))
        if not parts:
            continue
        title = ITEMS[item][0] if item in ITEMS else "Cover and front matter"
        sections.append(
            Section(
                item=item,
                title=title,
                page_start=parts[0][0],
                page_end=parts[-1][0],
                pages=tuple(parts),
            )
        )
    return sections
