from decimal import Decimal

import pytest
from pydantic import ValidationError

from tieout.agent.validation import (
    Citation,
    Evidence,
    EvidenceLedger,
    SubmitAnswerArgs,
    figures_in_answer,
    numbers_in,
    supported,
    validate_submission,
)

CHUNK = "ch_be9a964b06a211ba"
FACT = "xf_77403cb0139a90aa"


def ledger() -> EvidenceLedger:
    led = EvidenceLedger()
    text = "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)"
    led.add(Evidence(CHUNK, "chunk", text, numbers_in(text)))
    led.add(
        Evidence(
            FACT,
            "fact",
            "PaymentsToAcquirePropertyPlantAndEquipment USD FY2018: 1577000000",
            [Decimal(1577000000)],
        )
    )
    return led


def submit(answer: str, citations: list[tuple[str, str]], **kw: object) -> SubmitAnswerArgs:
    return SubmitAnswerArgs(
        answer=answer,
        confidence=0.9,
        rationale="r",
        citations=[Citation(source_id=s, quote=q) for s, q in citations],
        **kw,  # type: ignore[arg-type]
    )


def test_valid_answer_with_chunk_and_fact() -> None:
    args = submit(
        "3M's FY2018 capital expenditure was $1,577 million.",
        [
            (CHUNK, "Purchases of property, plant and equipment (PP&E) (1,577)"),
            (FACT, "1577000000"),
        ],
    )
    assert validate_submission(args, ledger()) == []


def test_unseen_or_made_up_source_is_rejected() -> None:
    args = submit("It was $1,577 million.", [("ch_0000000000000000", "1,577")])
    problems = validate_submission(args, ledger())
    assert any("not returned by any tool" in p for p in problems)


def test_quote_must_be_in_source() -> None:
    args = submit("It was $1,577 million.", [(CHUNK, "capital spending rose sharply")])
    assert any("does not appear" in p for p in validate_submission(args, ledger()))


def test_numeric_quote_formatting_is_flexible() -> None:
    args = submit("It was $1,577 million.", [(CHUNK, "1577")])
    assert validate_submission(args, ledger()) == []


def test_computed_number_without_calculator_is_rejected() -> None:
    # 1,577 - 1,373 = 204: correct, but the model did the math itself.
    args = submit("Capex rose by $204 million.", [(CHUNK, "(1,577) (1,373)")])
    problems = validate_submission(args, ledger())
    assert any("204" in p and "calculate" in p for p in problems)


def test_computed_number_from_calculator_is_accepted() -> None:
    led = ledger()
    led.add(Evidence("calc:1577-1373", "calc", "1577-1373", [Decimal(204)]))
    led.add(Evidence("calc:pct", "calc", "", [Decimal("14.857975")]))
    args = submit("Capex rose by $204 million (14.9%).", [(CHUNK, "(1,577) (1,373)")])
    assert validate_submission(args, led) == []


def test_citations_required_unless_needs_review() -> None:
    assert validate_submission(submit("Unknown.", []), ledger())
    assert validate_submission(submit("Not disclosed.", [], needs_review=True), ledger()) == []


def test_bad_source_id_format_fails_schema() -> None:
    with pytest.raises(ValidationError):
        Citation(source_id="page 60", quote="x")


@pytest.mark.parametrize(
    ("token", "value", "pool", "ok"),
    [
        ("1,577", Decimal(1577), [Decimal(1577000000)], True),  # millions
        ("1.6", Decimal("1.6"), [Decimal(1577000000)], True),  # rounded billions
        ("1.5", Decimal("1.5"), [Decimal(1577000000)], False),
        ("14.9%", Decimal("14.9"), [Decimal("0.14858")], True),  # percent vs ratio
        ("10.18", Decimal("10.18"), [Decimal("10.18")], True),
        ("10.25", Decimal("10.25"), [Decimal("10.18")], False),
    ],
)
def test_supported_scales_and_rounding(
    token: str, value: Decimal, pool: list[Decimal], ok: bool
) -> None:
    assert supported(token, value, pool) is ok


def test_years_counts_and_labels_are_not_figures() -> None:
    figures = [
        t for t, _ in figures_in_answer("In FY2018 (2018), 3M's 3 segments in Item 7 grew 4.5%")
    ]
    assert figures == ["4.5%"]


def test_trailing_punctuation_is_not_part_of_a_number() -> None:
    # Regression: "2015," was parsed as a figure and the answer was rejected.
    text = "In fiscal year 2015, Adobe's capex was $184.9 million ($184,936 thousand)."
    assert [t for t, _ in figures_in_answer(text)] == ["$184.9", "($184,936"]
    assert numbers_in("Revenue rose to 1,577, then 1,373.") == [Decimal(1577), Decimal(1373)]


def test_ledger_keeps_the_fullest_text_of_a_chunk() -> None:
    led = EvidenceLedger()
    led.add(
        Evidence(CHUNK, "chunk", "Current operating lease liabilities 4,134 3,859 Other cur…", [])
    )
    led.add(Evidence(CHUNK, "chunk", "short…", []))
    assert led.sources[CHUNK].text.startswith("Current operating lease")


def test_figures_from_the_question_are_allowed() -> None:
    # Validator v1 rejected this correct answer: "20%" comes from the question.
    args = submit(
        "No segment exceeds 20% of revenue; capex was $1,577 million.", [(CHUNK, "(1,577)")]
    )
    assert any("20%" in p for p in validate_submission(args, ledger()))
    question = "Does any segment represent more than 20% of revenue?"
    assert validate_submission(args, ledger(), question=question) == []


def test_dates_and_references_are_not_figures() -> None:
    # Regression: "December 31" and "Note 21" were treated as unsupported figures.
    text = (
        "Accounts payable as of December 31, 2019 were $1,380.5 million (see Note 21 and "
        "Item 7). Revenue was 31 million units in Dec. 31 quarter."
    )
    assert [t for t, _ in figures_in_answer(text)] == ["$1,380.5", "31"]
