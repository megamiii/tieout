from datetime import date
from decimal import Decimal
from typing import Any

from tieout.llm.types import ChatResponse, Message, Usage
from tieout.workflows.analytical import Change, explain, material_changes
from tieout.workflows.claims import extract_claims
from tieout.workflows.tieout import FactLite, classify, resolve, summarize

FY = (date(2022, 1, 1), date(2022, 12, 31))
FY_PRIOR = (date(2021, 1, 1), date(2021, 12, 31))


def fact(concept: str, value: int, period: tuple[date, date] = FY, fid: str = "") -> FactLite:
    return FactLite(
        fid or f"xf_{concept[:12].lower():0<16}", concept, concept, Decimal(value), *period
    )


def claim(text: str, year: int = 2022):
    [c] = extract_claims("ch_1", 40, text, year)
    return c


class Scripted:
    model = "fake"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, messages: list[Message], **kw: Any) -> ChatResponse:
        self.prompts.append(messages[0]["content"])
        return ChatResponse(self.replies.pop(0), [], "stop", Usage(10, 5), "fake", "h", False, 1.0)


RANK = ["NetIncomeLoss", "Revenues", "PaymentsToAcquirePropertyPlantAndEquipment", "Assets"]
FACTS = [
    fact("NetIncomeLoss", 5_349_000_000),
    fact("Revenues", 32_765_000_000),
    fact("PaymentsToAcquirePropertyPlantAndEquipment", 1_577_000_000),
    fact("Assets", 36_500_000_000, (None, date(2022, 12, 31))),  # type: ignore[arg-type]
    fact("NetIncomeLoss", 4_858_000_000, FY_PRIOR, "xf_prior0000000000"),
]


def test_match_within_written_rounding_and_year() -> None:
    r = classify(claim("Net income was $5.349 billion in 2022."), FACTS, RANK)
    assert r.status == "matched" and r.fact is not None and r.fact.concept == "NetIncomeLoss"
    r = classify(claim("Net income was $4.858 billion in 2021."), FACTS, RANK)
    assert r.status == "matched" and r.fact is not None and r.fact.fact_id == "xf_prior0000000000"


def test_value_match_on_unrelated_concept_is_not_a_match() -> None:
    # 1,577 million equals capex, but the sentence is about something ranked far away.
    r = classify(
        claim("Pension contributions were $1,577 million."), FACTS, ["UnrelatedConcept"] * 20
    )
    assert r.status == "unmatched" and "coincidence" in r.note


def test_quarterly_and_coarse_amounts_are_out_of_scope() -> None:
    assert (
        classify(claim("Fourth quarter sales were $8.3 billion."), FACTS, RANK).status
        == "out_of_scope"
    )
    coarse = classify(claim("Advances were $0.1 billion."), FACTS, RANK)
    assert coarse.status == "out_of_scope" and "coarsely" in coarse.note


def test_close_related_value_becomes_candidate_then_adjudicated() -> None:
    planted = claim("Net income was $5.1 billion in 2022.")  # XBRL says 5.349 billion
    r = classify(planted, FACTS, RANK)
    assert r.status == "candidate_mismatch" and r.candidates[0].concept == "NetIncomeLoss"

    llm = Scripted('{"same_quantity_fact": 1, "reasoning": "Both are 2022 net income."}')
    resolved = resolve(r, llm)
    assert resolved.status == "mismatched" and "differs by -249,000,000" in resolved.note
    assert "Net income was $5.1 billion" in llm.prompts[0]


def test_adjudicator_can_reject_a_candidate() -> None:
    r = classify(claim("Segment net income was $5.1 billion in 2022."), FACTS, RANK)
    llm = Scripted('{"same_quantity_fact": null, "reasoning": "A segment, not the company."}')
    resolved = resolve(r, llm)
    assert resolved.status == "unmatched" and resolved.adjudication == "A segment, not the company."


def test_without_an_llm_candidates_are_not_reported_as_mismatches() -> None:
    r = resolve(classify(claim("Net income was $5.1 billion in 2022."), FACTS, RANK), None)
    assert r.status == "unmatched" and "not adjudicated" in r.note


def test_summary_counts() -> None:
    results = [classify(claim("Net income was $5.349 billion in 2022."), FACTS, RANK)]
    assert summarize(results)["matched"] == 1


# --- analytical review ---------------------------------------------------------


def test_material_changes_use_threshold_and_materiality() -> None:
    facts = [
        fact("Revenues", 10_000_000_000),
        fact("Revenues", 9_000_000_000, FY_PRIOR),  # +11%: below 20%
        fact("ResearchAndDevelopmentExpense", 2_000_000_000),
        fact("ResearchAndDevelopmentExpense", 1_000_000_000, FY_PRIOR),  # +100%, material
        fact("OtherExpenses", 50_000_000),
        fact("OtherExpenses", 10_000_000, FY_PRIOR),  # +400% but < 1% of revenue
        fact("IncreaseDecreaseInInventories", 900_000_000),
        fact("IncreaseDecreaseInInventories", 100_000_000, FY_PRIOR),  # skipped by design
    ]
    changes, base = material_changes(facts, 2022)
    assert base == "Revenues"
    assert [c.concept for c in changes] == ["ResearchAndDevelopmentExpense"]
    assert changes[0].pct == 100


PASSAGES = [
    {
        "chunk_id": "ch_a",
        "text": "R&D expense increased primarily due to the Xilinx acquisition.",
        "page": 44,
    },
    {"chunk_id": "ch_b", "text": "Revenue grew in all segments.", "page": 43},
]
CHANGE = Change(
    "ResearchAndDevelopmentExpense",
    "R&D",
    fact("ResearchAndDevelopmentExpense", 2_000_000_000),
    fact("ResearchAndDevelopmentExpense", 1_000_000_000, FY_PRIOR),
)


def test_explanation_quote_is_verified() -> None:
    good = Scripted(
        '{"explained": true, "explanation": "Acquisition.", "chunk_id": "ch_a", '
        '"quote": "increased primarily due to the Xilinx acquisition"}'
    )
    assert explain(good, CHANGE, PASSAGES, 2022).status == "explained"
    made_up = Scripted(
        '{"explained": true, "explanation": "Hiring.", "chunk_id": "ch_a", '
        '"quote": "due to hiring"}'
    )
    assert explain(made_up, CHANGE, PASSAGES, 2022).status == "unverified"
    wrong_chunk = Scripted(
        '{"explained": true, "explanation": "x", "chunk_id": "ch_z", "quote": "Revenue grew"}'
    )
    assert explain(wrong_chunk, CHANGE, PASSAGES, 2022).status == "unverified"


def test_no_explanation() -> None:
    none = Scripted('{"explained": false, "explanation": "", "chunk_id": null, "quote": null}')
    assert explain(none, CHANGE, PASSAGES, 2022).status == "not_explained"
