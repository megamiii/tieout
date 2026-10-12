"""Analytical review: material year-over-year changes and management's explanation.

1. From the filing's own XBRL facts, compare each line item's value for the
   fiscal year with the prior year (both as reported in this 10-K). A change
   is material if it is at least MIN_CHANGE (20%) and the line item is at
   least MATERIALITY (1%) of revenue (total assets for companies without a
   revenue concept). The largest changes in dollars are reviewed.
2. For each one, the filing's text is searched for passages about it, and the
   LLM is asked to quote management's explanation from those passages. The
   quote is checked against the passage it cites; a missing or unverifiable
   quote means "no explanation found", which is itself an audit finding.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from tieout.agent.validation import normalize
from tieout.evals.grading import ChatModel
from tieout.workflows.tieout import FactLite

MIN_CHANGE = Decimal("0.20")
MATERIALITY = Decimal("0.01")
MAX_ITEMS = 8
REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenuesNetOfInterestExpense",
)
# Skipped line items: cash-flow "change in X" lines restate balance-sheet
# movements; balance-sheet identities and equity / comprehensive-income
# components move with other items (one acquisition moved AMD's assets, total
# liabilities and equity, equity and paid-in capital) and are rarely explained
# on their own.
SKIP_PREFIXES = (
    "IncreaseDecrease",
    "LiabilitiesAndStockholdersEquity",
    "StockholdersEquity",
    "AdditionalPaidInCapital",
    "RetainedEarnings",
    "TreasuryStock",
    "ComprehensiveIncome",
    "OtherComprehensiveIncome",
    "AccumulatedOtherComprehensiveIncome",
    "CommonStockValue",
)


@dataclass
class Change:
    concept: str
    label: str
    current: FactLite
    prior: FactLite

    @property
    def delta(self) -> Decimal:
        return self.current.value - self.prior.value

    @property
    def pct(self) -> Decimal:
        return self.delta / abs(self.prior.value) * 100


def materiality_base(facts: Sequence[FactLite], fiscal_year: int) -> tuple[str, Decimal] | None:
    for concepts in (REVENUE_CONCEPTS, ("Assets",)):
        for concept in concepts:
            for f in facts:
                if f.concept == concept and f.fiscal_year == fiscal_year and f.annual_or_instant:
                    return concept, abs(f.value)
    return None


def material_changes(
    facts: Sequence[FactLite], fiscal_year: int, max_items: int = MAX_ITEMS
) -> tuple[list[Change], str | None]:
    """Largest material YoY changes, and the concept used as the materiality base."""
    base = materiality_base(facts, fiscal_year)
    if base is None:
        return [], None
    base_concept, base_value = base
    threshold = base_value * MATERIALITY
    by_key: dict[tuple[str, int], FactLite] = {}
    for f in facts:
        if not f.annual_or_instant or f.concept.startswith(SKIP_PREFIXES):
            continue
        by_key.setdefault((f.concept, f.fiscal_year), f)
    changes = []
    for (concept, year), cur in by_key.items():
        if year != fiscal_year:
            continue
        prior = by_key.get((concept, fiscal_year - 1))
        if prior is None or prior.value == 0:
            continue
        # Instants and durations must not be compared with each other.
        if (cur.period_start is None) != (prior.period_start is None):
            continue
        change = Change(concept, cur.label, cur, prior)
        size = max(abs(cur.value), abs(prior.value))
        if size >= threshold and abs(change.pct) >= MIN_CHANGE * 100:
            changes.append(change)
    changes.sort(key=lambda c: (-abs(c.delta), c.concept))
    # Two concepts with identical values for both years are the same movement.
    unique: list[Change] = []
    seen: set[tuple[Decimal, Decimal]] = set()
    for c in changes:
        key = (c.current.value, c.prior.value)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:max_items], base_concept


def explanation_query(change: Change) -> str:
    """Search query: the line item in plain words plus the figures, which MD&A
    explanations usually restate ("net revenue of $23.6 billion increased 44%")."""
    label = re.sub(r"\s*\([^)]*\)", "", change.label)
    label = re.sub(r",?\s*(Excluding Assessed Tax|Net of Tax|Attributable to Parent)", "", label)
    figures = []
    for v in (abs(change.current.value), abs(change.prior.value)):
        figures.append(f"${v / 10**9:.1f} billion" if v >= 10**9 else f"${v / 10**6:,.0f} million")
    return f"{label} increased decreased primarily due to {' '.join(figures)}"


EXPLAIN_PROMPT = """\
An auditor is reviewing a material change in a company's 10-K.

Line item: {label} ({concept})
Fiscal year {year}: {current:,.0f} USD; prior year: {prior:,.0f} USD ({pct:+.1f}%).

Passages from the same 10-K:
{passages}

Does management explain why this line item changed? Use only these passages. If one \
explains it, quote the explaining text exactly (under 300 characters) and give its \
chunk_id. If none does, say so; do not guess.

Reply with only a JSON object: {{"explained": true or false, "explanation": "<one \
sentence summary>", "chunk_id": "<chunk_id or null>", "quote": "<exact quote or null>"}}"""

ExplanationStatus = Literal["explained", "not_explained", "unverified"]


@dataclass
class Explanation:
    status: ExplanationStatus
    summary: str
    chunk_id: str | None = None
    quote: str | None = None


def explain(
    llm: ChatModel, change: Change, passages: Sequence[dict[str, Any]], year: int
) -> Explanation:
    """Ask for management's explanation; verify the quote is in the cited passage."""
    text_by_id = {p["chunk_id"]: p["text"] for p in passages}
    shown = "\n\n".join(
        f"[{p['chunk_id']}] ({p.get('section', '')}, p.{p.get('page', '?')})\n{p['text']}"
        for p in passages
    )
    prompt = EXPLAIN_PROMPT.format(
        label=change.label,
        concept=change.concept,
        year=year,
        current=abs(change.current.value),
        prior=abs(change.prior.value),
        pct=change.pct,
        passages=shown,
    )
    resp = llm.chat([{"role": "user", "content": prompt}], max_tokens=500)
    match = re.search(r"\{.*\}", resp.content or "", re.DOTALL)
    try:
        data: dict[str, Any] = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    summary = str(data.get("explanation") or "")
    if not data.get("explained"):
        return Explanation("not_explained", summary or "no explanation in the passages")
    chunk_id, quote = data.get("chunk_id"), data.get("quote")
    source = text_by_id.get(str(chunk_id)) if chunk_id else None
    if not source or not quote or normalize(str(quote)) not in normalize(source):
        return Explanation(
            "unverified", summary, str(chunk_id) if chunk_id else None, str(quote or "")
        )
    return Explanation("explained", summary, str(chunk_id), str(quote))
