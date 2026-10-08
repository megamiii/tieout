"""Evidence collected during a run, and validation of the final answer.

Every tool result registers the sources it showed the model (chunks, XBRL
facts) and every calculation result. submit_answer is validated only against
this run's evidence:

1. each cited source_id must have been returned by a tool in this run
   (catches made-up IDs and real IDs the model never saw);
2. each citation's quote must appear in that source (text match for chunks,
   numeric match at any scale for facts);
3. every number in the answer must come from a cited source or a calculate
   result: the model may copy and round numbers, never compute them.

No database is needed, so validation works the same live and on replay.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["chunk", "fact", "calc"]

# Scales an answer may use relative to a source: $1,577 million vs 1577000000,
# 14.9% vs 0.149.
SCALES = (Decimal(1), Decimal(10) ** 3, Decimal(10) ** 6, Decimal(10) ** 9, Decimal("0.01"))
RELATIVE_TOLERANCE = Decimal("0.005")
SMALL_INT_MAX = 12  # "3 segments", "Item 7": counts and labels, not figures

# A thousands separator must be followed by exactly three digits, so the comma
# in "fiscal year 2015, Adobe" is punctuation, not part of the number.
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.,])\(?-?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?%?(?![A-Za-z0-9])"
)


@dataclass
class Evidence:
    source_id: str
    kind: SourceKind
    text: str
    values: list[Decimal]
    meta: dict[str, Any] = field(default_factory=dict)


class EvidenceLedger:
    def __init__(self) -> None:
        self.sources: dict[str, Evidence] = {}
        self.calc_results: list[Decimal] = []

    def add(self, evidence: Evidence) -> None:
        if evidence.kind == "calc":
            self.calc_results.extend(evidence.values)
        else:
            self.sources[evidence.source_id] = evidence


def to_decimal(token: str) -> Decimal | None:
    cleaned = token.strip("()$%").replace(",", "").replace("$", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def numbers_in(text: str) -> list[Decimal]:
    out = []
    for m in NUMBER_RE.finditer(text):
        d = to_decimal(m.group(0))
        if d is not None:
            out.append(d)
    return out


def figures_in_answer(text: str) -> list[tuple[str, Decimal]]:
    """Numbers in an answer that must be supported: not years, not small counts."""
    out = []
    for m in NUMBER_RE.finditer(text):
        token = m.group(0)
        d = to_decimal(token)
        if d is None:
            continue
        plain = token.strip("()$%-")
        is_year = re.fullmatch(r"(19|20)\d\d", plain) is not None
        is_small_count = "." not in plain and abs(d) <= SMALL_INT_MAX and "%" not in token
        if not is_year and not is_small_count:
            out.append((token, d))
    return out


def decimals_of(token: str) -> int:
    plain = token.strip("()$%").replace(",", "")
    return len(plain.split(".")[1]) if "." in plain else 0


def supported(token: str, value: Decimal, pool: list[Decimal]) -> bool:
    """Does `value` (as written in `token`) match some pool value at some scale?

    Tolerance allows rounding: "1.6 billion" matches 1,577,000,000 because 1.6
    has one decimal place (±0.05 billion), and 0.5% relative error otherwise.
    """
    x = abs(value)
    rounding = Decimal("0.5") * Decimal(10) ** -decimals_of(token)
    for y in pool:
        y = abs(y)
        for scale in SCALES:
            tol = max(y * RELATIVE_TOLERANCE, rounding * scale)
            if abs(x * scale - y) <= tol:
                return True
    return False


def normalize(text: str) -> str:
    text = text.lower().replace("\u2019", "'").replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,577 -> 1577
    return re.sub(r"\s+", " ", text).strip()


class Citation(BaseModel):
    source_id: str = Field(
        pattern=r"^(ch|xf)_[0-9a-f]{16}$",
        description="A chunk_id (ch_...) or XBRL fact_id (xf_...) returned by a tool.",
    )
    quote: str = Field(
        min_length=1,
        max_length=300,
        description="Exact text (or the number) from that source that supports the answer.",
    )


class SubmitAnswerArgs(BaseModel):
    answer: str = Field(min_length=1, max_length=2000, description="The final answer.")
    confidence: float = Field(ge=0, le=1, description="Probability the answer is correct.")
    rationale: str = Field(max_length=2000, description="How the sources support the answer.")
    citations: list[Citation] = Field(default_factory=list, max_length=10)
    needs_review: bool = Field(
        default=False,
        description="True if evidence is missing, ambiguous or conflicting.",
    )
    review_reason: str | None = Field(default=None, max_length=500)


def validate_submission(
    args: SubmitAnswerArgs, ledger: EvidenceLedger, *, check_figures: bool = True
) -> list[str]:
    """Problems with a submission; an empty list means it is accepted.

    `check_figures` enforces rule 3; it is off when the agent has no
    calculate tool (an ablation), since it could not satisfy it.
    """
    problems: list[str] = []
    if not args.citations and not args.needs_review:
        problems.append("cite at least one source (or set needs_review if there is none)")

    cited: list[Evidence] = []
    for c in args.citations:
        ev = ledger.sources.get(c.source_id)
        if ev is None:
            problems.append(
                f"{c.source_id} was not returned by any tool in this run; "
                "cite only chunk_ids/fact_ids you received"
            )
            continue
        cited.append(ev)
        if not quote_in_source(c.quote, ev):
            problems.append(
                f"quote {c.quote!r} does not appear in {c.source_id}; quote the source exactly"
            )

    pool = [v for ev in cited for v in ev.values] + ledger.calc_results
    for token, value in figures_in_answer(args.answer) if check_figures else []:
        if not supported(token, value, pool):
            problems.append(
                f"the number {token} is not in any cited source or calculate result; "
                "cite its source, or compute it with the calculate tool"
            )
    return problems


def quote_in_source(quote: str, ev: Evidence) -> bool:
    if ev.kind == "fact":
        if normalize(quote) in normalize(ev.text):
            return True
        nums = numbers_in(quote)
        return bool(nums) and all(supported(str(n), n, ev.values) for n in nums)
    if normalize(quote) in normalize(ev.text):
        return True
    # A purely numeric quote may be formatted differently ("(1,577)" vs "1577").
    nums = numbers_in(quote)
    stripped = re.sub(r"[\d,.$%()\-\s]", "", quote)
    return bool(nums) and not stripped and all(n in ev.values or -n in ev.values for n in nums)
