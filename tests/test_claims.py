from decimal import Decimal

from tieout.workflows.claims import (
    assign_years,
    dedupe,
    extract_claims,
    parse_amount,
    split_sentences,
)

NIKE = (
    "In fiscal 2023, NIKE, Inc. achieved record Revenues of $51.2 billion, which increased 10%. "
    "NIKE Direct revenues grew 14% from $18.7 billion in fiscal 2022 "
    "to $21.3 billion in fiscal 2023."
)
AXP = (
    "Represents net income, less earnings allocated to participating share awards of $57 million, "
    "$56 million and $20 million for the years ended December 31, 2022, 2021 and 2020, "
    "respectively."
)
MMM = (
    "For the fourth quarter of 2018, net income attributable to 3M was $1.347 billion, "
    "or $2.27 per diluted share. Adjusted net income (non-GAAP) was $1.366 billion."
)


def test_parse_amount_keeps_written_precision() -> None:
    assert parse_amount("1", ".347", "billion") == (Decimal("1347000000"), Decimal("500000"))
    assert parse_amount("51", ".2", "billion") == (Decimal("51200000000"), Decimal("50000000"))
    assert parse_amount("1,577", None, "million") == (Decimal("1577000000"), Decimal("500000"))


def test_sentences_do_not_split_on_decimals() -> None:
    assert len(split_sentences(NIKE)) == 2


def test_years_follow_the_amount() -> None:
    claims = extract_claims("ch_1", 33, NIKE, default_year=2023)
    assert [(c.amount_text, c.year) for c in claims] == [
        ("$51.2 billion", 2023),
        ("$18.7 billion", 2022),
        ("$21.3 billion", 2023),
    ]


def test_respectively_maps_years_in_order() -> None:
    claims = extract_claims("ch_2", 50, AXP, default_year=2022)
    assert [(c.amount_text, c.year) for c in claims] == [
        ("$57 million", 2022),
        ("$56 million", 2021),
        ("$20 million", 2020),
    ]


def test_quarterly_and_non_gaap_flags() -> None:
    first, second = extract_claims("ch_3", 15, MMM, default_year=2018)
    assert first.quarterly and not first.non_gaap and first.year == 2018
    assert second.non_gaap and not second.quarterly
    # "$2.27 per diluted share" has no scale word: not a claim
    assert "$2.27" not in [c.amount_text for c in (first, second)]


def test_default_year_when_none_stated() -> None:
    [c] = extract_claims("ch_4", 20, "Capital spending was $1.6 billion.", default_year=2018)
    assert c.year == 2018


def test_assign_years_falls_back_to_preceding_year() -> None:
    s = "In 2021 the company spent $5 million on research."
    assert assign_years(s, [(26, 36)], None) == [2021]


def test_overlapping_chunks_are_deduplicated() -> None:
    full = extract_claims("ch_a", 33, NIKE, 2023)
    overlap = extract_claims("ch_b", 33, NIKE.split(". ", 1)[1], 2023)  # second sentence again
    merged = dedupe(full + overlap)
    assert len(merged) == 3


def test_same_amount_in_different_sentences_is_kept() -> None:
    text = "Capex was $1.6 billion in 2022. Separately, dividends were $1.6 billion in 2022."
    assert len(dedupe(extract_claims("ch_5", 20, text, 2022))) == 2


def test_year_after_an_amount_belongs_to_the_next_amount() -> None:
    # Regression: "$5.1 billion" was assigned 2021, the year of the next amount.
    s = "Net loss for 2022 was $5.1 billion compared with net loss of $4.3 billion in 2021."
    claims = extract_claims("ch_6", 30, s, default_year=2022)
    assert [(c.amount_text, c.year) for c in claims] == [
        ("$5.1 billion", 2022),
        ("$4.3 billion", 2021),
    ]
