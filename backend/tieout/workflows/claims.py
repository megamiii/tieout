"""Extract numeric claims from MD&A prose: "$1.6 billion", "$523 million".

Only dollar amounts with an explicit scale word are claims: in prose they are
unambiguous, whereas bare numbers in tables depend on a column header ("in
millions") that is lost in text extraction. Each claim keeps the precision it
was written with, so matching can allow exactly the rounding the author used:
"$1.6 billion" means 1.55-1.65 billion, "$1,577 million" means 1,576.5-1,577.5
million.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal

SCALES = {"thousand": Decimal(10) ** 3, "million": Decimal(10) ** 6, "billion": Decimal(10) ** 9}

AMOUNT_RE = re.compile(
    r"\$\s?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?\s*(?P<scale>thousand|million|billion)\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"(?<!\d)(?:fiscal\s+(?:year\s+)?)?((?:19|20)\d\d)(?!\d)", re.IGNORECASE)
SENTENCE_END_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(\u2022\u25cf\u2013-])")
QUARTER_RE = re.compile(r"\b(?:quarter|three months|six months|nine months|Q[1-4])\b", re.I)
NON_GAAP_RE = re.compile(
    r"\b(?:non-GAAP|adjusted|organic|currency-neutral|constant currency)\b", re.I
)
CONTEXT_CHARS = 120


@dataclass
class Claim:
    chunk_id: str
    page: int
    sentence: str
    amount_text: str  # as written, e.g. "$1.347 billion"
    value: Decimal  # in dollars
    tolerance: Decimal  # half a unit of the last digit written, in dollars
    year: int | None  # fiscal year the amount refers to, if stated
    quarterly: bool  # refers to a quarter or interim period
    non_gaap: bool  # sentence describes a non-GAAP / adjusted measure
    start: int  # offset of the amount within the sentence
    flags: list[str] = field(default_factory=list)

    @property
    def context(self) -> str:
        """Text around the amount, used to judge what it measures."""
        a = max(0, self.start - CONTEXT_CHARS)
        return self.sentence[a : self.start + len(self.amount_text) + 40]


def parse_amount(num: str, dec: str | None, scale: str) -> tuple[Decimal, Decimal]:
    """Value and rounding tolerance in dollars."""
    mult = SCALES[scale.lower()]
    digits = num.replace(",", "") + (dec or "")
    value = Decimal(digits) * mult
    places = len(dec) - 1 if dec else 0
    tolerance = Decimal("0.5") * Decimal(10) ** -places * mult
    return value, tolerance


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return [s for s in SENTENCE_END_RE.split(flat) if s]


def assign_years(
    sentence: str, spans: list[tuple[int, int]], default: int | None
) -> list[int | None]:
    """Fiscal year of each amount in a sentence.

    - "A, B and C for 2022, 2021 and 2020, respectively": in order.
    - otherwise a year right after the amount, before the next amount
      ("$21.3 billion in fiscal 2023"); else the nearest year before it
      ("for 2022 was $5.1 billion compared with $4.3 billion in 2021": 2022,
      since 2021 belongs to the next amount); else `default` (the filing's year).
    """
    years = [(m.start(), int(m.group(1))) for m in YEAR_RE.finditer(sentence)]
    if "respectively" in sentence.lower():
        after_last = [y for pos, y in years if pos > spans[-1][1]]
        if len(after_last) == len(spans):
            return list(after_last)
    out: list[int | None] = []
    for i, (start, end) in enumerate(spans):
        next_start = spans[i + 1][0] if i + 1 < len(spans) else len(sentence)
        following = [y for pos, y in years if end <= pos < min(next_start, end + 60)]
        preceding = [y for pos, y in years if pos < start]
        out.append(following[0] if following else preceding[-1] if preceding else default)
    return out


def extract_claims(chunk_id: str, page: int, text: str, default_year: int | None) -> list[Claim]:
    claims = []
    for sentence in split_sentences(text):
        matches = list(AMOUNT_RE.finditer(sentence))
        if not matches:
            continue
        years = assign_years(sentence, [m.span() for m in matches], default_year)
        quarterly = QUARTER_RE.search(sentence) is not None
        non_gaap = NON_GAAP_RE.search(sentence) is not None
        for m, year in zip(matches, years, strict=True):
            value, tol = parse_amount(m.group("num"), m.group("dec"), m.group("scale"))
            claims.append(
                Claim(
                    chunk_id=chunk_id,
                    page=page,
                    sentence=sentence,
                    amount_text=m.group(0),
                    value=value,
                    tolerance=tol,
                    year=year,
                    quarterly=quarterly,
                    non_gaap=non_gaap,
                    start=m.start(),
                )
            )
    return claims


def _same_claim(a: Claim, b: Claim) -> bool:
    """One sentence seen in two overlapping chunks: the copy cut at a chunk
    boundary is a substring of the full sentence, with the amount at the same
    position. Two different sentences that merely share an amount are not."""
    if (a.page, a.amount_text.lower()) != (b.page, b.amount_text.lower()):
        return False
    long, short = (a, b) if len(a.sentence) >= len(b.sentence) else (b, a)
    offset = long.sentence.find(short.sentence)
    while offset != -1:
        if offset + short.start == long.start:
            return True
        offset = long.sentence.find(short.sentence, offset + 1)
    return False


def dedupe(claims: list[Claim]) -> list[Claim]:
    """Chunks overlap, so one sentence can be extracted twice; keep the fuller copy."""
    kept: list[Claim] = []
    for c in sorted(claims, key=lambda c: -len(c.sentence)):
        if not any(_same_claim(c, k) for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: (c.page, c.chunk_id, c.start))
