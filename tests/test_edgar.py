from datetime import date
from decimal import Decimal

import pytest

from tieout.edgar.client import FilingRef
from tieout.edgar.filings import FilingMatchError, fiscal_year_label, match_annual_filing
from tieout.edgar.xbrl import iter_facts, make_fact_id


def ref(acc: str, form: str, filed: str, period: str) -> FilingRef:
    return FilingRef(acc, form, date.fromisoformat(filed), date.fromisoformat(period), "d.htm")


@pytest.mark.parametrize(
    ("period_end", "label"),
    [
        ("2022-12-31", 2022),  # calendar year
        ("2023-05-31", 2023),  # Nike: FY ends May
        ("2023-01-28", 2023),  # Best Buy: FY2023 ends late January 2023
        ("2021-01-02", 2020),  # 52/53-week year ending in the first days of January
    ],
)
def test_fiscal_year_label(period_end: str, label: int) -> None:
    assert fiscal_year_label(date.fromisoformat(period_end)) == label


FILINGS = [
    ref("0000000001-21-000001", "10-K", "2021-03-19", "2021-01-30"),
    ref("0000000001-22-000001", "10-K", "2022-03-18", "2022-01-29"),
    ref("0000000001-22-000009", "10-K/A", "2022-05-01", "2022-01-29"),
]


def test_match_prefers_original_over_amendment() -> None:
    assert match_annual_filing(FILINGS, 2022).accession == "0000000001-22-000001"


def test_match_uses_period_not_sec_fy() -> None:
    # The SEC's own `fy` for this filing is 2020; the period end says FY2021.
    assert match_annual_filing(FILINGS, 2021).accession == "0000000001-21-000001"


def test_match_rejects_contradicting_link() -> None:
    link = "https://example.com/content/0000000001-22-000009/doc.pdf"
    with pytest.raises(FilingMatchError):
        match_annual_filing(FILINGS, 2022, link_hint=link)


def test_match_missing_year() -> None:
    with pytest.raises(FilingMatchError):
        match_annual_filing(FILINGS, 2019)


COMPANYFACTS = {
    "cik": 66740,
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        # duration fact from a 10-K
                        {
                            "start": "2022-01-01",
                            "end": "2022-12-31",
                            "val": 34229000000,
                            "accn": "0000066740-23-000014",
                            "fy": 2022,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-02-08",
                            "frame": "CY2022",
                        },
                        # same fact repeated (as companyfacts sometimes does): dropped
                        {
                            "start": "2022-01-01",
                            "end": "2022-12-31",
                            "val": 34229000000,
                            "accn": "0000066740-23-000014",
                            "fy": 2022,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-02-08",
                        },
                        # 10-Q fact: filtered out
                        {
                            "start": "2022-01-01",
                            "end": "2022-03-31",
                            "val": 8829000000,
                            "accn": "0000066740-22-000040",
                            "fy": 2022,
                            "fp": "Q1",
                            "form": "10-Q",
                            "filed": "2022-04-26",
                        },
                    ]
                },
            },
            "EarningsPerShareDiluted": {
                "label": "EPS diluted",
                "units": {
                    "USD/shares": [
                        {
                            "start": "2022-01-01",
                            "end": "2022-12-31",
                            "val": 10.18,
                            "accn": "0000066740-23-000014",
                            "fy": 2022,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-02-08",
                        },
                    ]
                },
            },
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "label": "Shares outstanding",
                "units": {
                    "shares": [
                        {
                            "end": "2023-01-31",
                            "val": 550000000,
                            "accn": "0000066740-23-000014",
                            "fy": 2022,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-02-08",
                        },
                    ]
                },
            }
        },
    },
}


def test_iter_facts_filters_and_dedupes() -> None:
    facts = list(iter_facts(COMPANYFACTS))
    assert [f.concept for f in facts] == [
        "Revenues",
        "EarningsPerShareDiluted",
        "EntityCommonStockSharesOutstanding",
    ]
    revenue, eps, shares = facts
    assert revenue.value == Decimal(34229000000)
    assert revenue.duration_days == 364
    # floats are converted exactly as written, not via binary float error
    assert eps.value == Decimal("10.18")
    assert shares.period_start is None and shares.duration_days is None


def test_fact_id_is_stable() -> None:
    args = (66740, "us-gaap", "Revenues", "USD", date(2022, 1, 1), date(2022, 12, 31), "acc")
    assert make_fact_id(*args) == make_fact_id(*args)
    assert make_fact_id(*args).startswith("xf_") and len(make_fact_id(*args)) == 19
    other = (66740, "us-gaap", "Revenues", "USD", date(2022, 1, 1), date(2022, 12, 31), "acc2")
    assert make_fact_id(*args) != make_fact_id(*other)
