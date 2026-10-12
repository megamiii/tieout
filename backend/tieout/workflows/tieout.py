"""Tie-out: check MD&A dollar claims against the filing's own XBRL facts.

For each claim (see claims.py), against the facts reported in the same 10-K:

1. Deterministic screening
   - out of scope: quarterly/interim amounts (a 10-K's XBRL is annual), and
     amounts rounded so coarsely that tying them out is meaningless
     ("$0.1 billion" = 50-150 million).
   - matched: a fact equals the claim within the claim's own rounding, for the
     fiscal year the claim refers to, and the fact's concept is related to the
     sentence (ranks in the top MATCH_RANK of the filing's concepts by
     similarity to the claim's context). Value alone isn't enough: "$8
     million" sits within rounding of several unrelated facts by chance.
   - candidate mismatch: no related fact has the value, but one of the
     TOP_RELATED most similar concepts has a close value (within
     MISMATCH_RATIO) for that year.
   - unmatched: anything else (segment figures, non-GAAP measures, untagged
     amounts), noting when only an unrelated fact matched the value.

2. LLM adjudication of candidate mismatches
   Embedding similarity on concept labels can't tell "Health Care segment
   sales" from total SG&A: on five filings, a spot check of 25 unadjudicated
   candidates found almost all were different quantities. So each candidate
   goes to the LLM, which must say whether the sentence's amount measures the
   same quantity, scope (company vs segment) and period as one of the facts.
   Only a confirmed same quantity with a different value is "mismatched".
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from tieout.edgar.filings import fiscal_year_label
from tieout.evals.grading import ChatModel
from tieout.workflows.claims import Claim

MATCH_RANK = 15
TOP_RELATED = 3
MISMATCH_RATIO = (Decimal("0.8"), Decimal("1.25"))
MAX_RELATIVE_ROUNDING = Decimal("0.05")
ANNUAL_DAYS = (350, 380)

Status = Literal["matched", "candidate_mismatch", "mismatched", "unmatched", "out_of_scope"]
Vec = npt.NDArray[np.float32]


@dataclass(frozen=True)
class FactLite:
    fact_id: str
    concept: str
    label: str
    value: Decimal
    period_start: date | None
    period_end: date

    @property
    def fiscal_year(self) -> int:
        return fiscal_year_label(self.period_end)

    @property
    def annual_or_instant(self) -> bool:
        if self.period_start is None:
            return True
        return ANNUAL_DAYS[0] <= (self.period_end - self.period_start).days <= ANNUAL_DAYS[1]

    @property
    def period(self) -> str:
        if self.period_start is None:
            return f"as of {self.period_end}"
        return f"{self.period_start} to {self.period_end}"


@dataclass
class ClaimResult:
    claim: Claim
    status: Status
    fact: FactLite | None = None  # matched fact, or the same-quantity fact that disagrees
    concept_rank: int | None = None
    note: str = ""
    candidates: list[FactLite] = field(default_factory=list)  # for adjudication
    adjudication: str = ""  # the LLM's reasoning, when adjudicated


def concept_ranking(
    claim_vec: Vec, concepts: Sequence[str], concept_vecs: dict[str, Vec]
) -> list[str]:
    """Concepts ordered by similarity to the claim's context, most similar first."""
    scored = [(float(concept_vecs[c] @ claim_vec), c) for c in concepts if c in concept_vecs]
    return [c for _, c in sorted(scored, key=lambda t: (-t[0], t[1]))]


def classify(claim: Claim, facts: Sequence[FactLite], ranking: list[str]) -> ClaimResult:
    """Deterministic screening (step 1)."""
    if claim.quarterly:
        return ClaimResult(claim, "out_of_scope", note="quarterly or interim amount")
    if claim.tolerance > claim.value * MAX_RELATIVE_ROUNDING:
        return ClaimResult(claim, "out_of_scope", note="too coarsely rounded to tie out")
    rank_of = {c: i + 1 for i, c in enumerate(ranking)}
    usable = [f for f in facts if f.annual_or_instant]
    same_year = [f for f in usable if claim.year is None or f.fiscal_year == claim.year]

    value_matches = [f for f in same_year if abs(abs(f.value) - claim.value) <= claim.tolerance]
    related = [f for f in value_matches if rank_of.get(f.concept, 10**6) <= MATCH_RANK]
    if related:
        best = min(related, key=lambda f: (rank_of[f.concept], f.fact_id))
        return ClaimResult(claim, "matched", best, rank_of[best.concept])

    top = set(ranking[:TOP_RELATED])
    close = [
        f
        for f in same_year
        if f.concept in top
        and f.value != 0
        and MISMATCH_RATIO[0] <= claim.value / abs(f.value) <= MISMATCH_RATIO[1]
    ]
    if close:
        close.sort(key=lambda f: (rank_of[f.concept], f.fact_id))
        return ClaimResult(claim, "candidate_mismatch", candidates=close)

    if value_matches:
        f = min(value_matches, key=lambda f: (rank_of.get(f.concept, 10**6), f.fact_id))
        return ClaimResult(
            claim,
            "unmatched",
            note=f"value equals {f.concept} (concept rank {rank_of.get(f.concept, '-')}), "
            "an unrelated concept: likely coincidence",
        )
    reason = "non-GAAP / adjusted measure" if claim.non_gaap else "not found in XBRL"
    return ClaimResult(claim, "unmatched", note=reason)


ADJUDICATION_PROMPT = """\
You check a 10-K's MD&A text against the company's XBRL financial data.

Sentence (fiscal year {year}): "{sentence}"
Amount in question: {amount}

XBRL facts reported in the same 10-K:
{facts}

Does the amount {amount} in the sentence state the same financial quantity as one \
of these facts: the same measure, the same scope (whole company, not a segment, \
product line or sub-component) and the same period? A change, difference, \
subtotal or related-but-different measure is not the same quantity.

Reply with only a JSON object: {{"same_quantity_fact": <fact number or null>, \
"reasoning": "<one sentence>"}}"""


def adjudicate(
    llm: ChatModel, claim: Claim, candidates: Sequence[FactLite]
) -> tuple[int | None, str]:
    """Index of the candidate that measures the same quantity, or None, plus reasoning."""
    facts = "\n".join(
        f"{i + 1}. {f.label} ({f.concept}), {f.period}: {abs(f.value):,.0f} USD"
        for i, f in enumerate(candidates)
    )
    prompt = ADJUDICATION_PROMPT.format(
        year=claim.year or "unknown", sentence=claim.sentence, amount=claim.amount_text, facts=facts
    )
    resp = llm.chat([{"role": "user", "content": prompt}], max_tokens=300)
    text = resp.content or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data: dict[str, Any] = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    reasoning = str(data.get("reasoning") or text[:300])
    choice = data.get("same_quantity_fact")
    if isinstance(choice, int) and 1 <= choice <= len(candidates):
        return choice - 1, reasoning
    return None, reasoning


def resolve(result: ClaimResult, llm: ChatModel | None) -> ClaimResult:
    """Turn a candidate mismatch into mismatched/unmatched (step 2)."""
    if result.status != "candidate_mismatch":
        return result
    if llm is None:
        result.status = "unmatched"
        result.note = "possible mismatch, not adjudicated (no LLM configured)"
        return result
    index, reasoning = adjudicate(llm, result.claim, result.candidates)
    result.adjudication = reasoning
    if index is None:
        result.status = "unmatched"
        result.note = "a related fact has a close value, but measures a different quantity"
        return result
    fact = result.candidates[index]
    result.fact = fact
    diff = result.claim.value - abs(fact.value)
    if abs(diff) <= result.claim.tolerance:  # defensive: screening already excluded this
        result.status = "matched"
        return result
    result.status = "mismatched"
    result.note = (
        f"XBRL reports {abs(fact.value):,.0f} USD; the text differs by {diff:+,.0f} "
        f"(beyond its rounding of ±{result.claim.tolerance:,.0f})"
    )
    return result


def summarize(results: Sequence[ClaimResult]) -> dict[str, int]:
    out = {"claims": len(results), "matched": 0, "mismatched": 0, "unmatched": 0, "out_of_scope": 0}
    for r in results:
        out[r.status if r.status != "candidate_mismatch" else "unmatched"] += 1
    return out
