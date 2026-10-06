"""XBRL fact selection against real Postgres (skipped if unavailable).

Synthetic facts are inserted in a transaction that is always rolled back.
"""

from collections.abc import Iterator
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from tieout.db.models import Company, XbrlFactRow
from tieout.db.session import get_engine, init_db
from tieout.edgar.facts_query import alias_concepts, annual_values, fiscal_year_ends
from tieout.edgar.xbrl import make_fact_id

pytestmark = pytest.mark.db
CIK = 9_999_999_002


@pytest.fixture
def session() -> Iterator[Session]:
    engine = get_engine()
    try:
        init_db(engine)
    except OperationalError:
        pytest.skip("Postgres not running (docker compose up -d)")
    conn = engine.connect()
    trans = conn.begin()
    s = Session(bind=conn)
    try:
        yield s
    finally:
        s.close()
        trans.rollback()
        conn.close()


def fact(
    concept: str, value: int, start: date | None, end: date, accession: str, filed: date
) -> XbrlFactRow:
    return XbrlFactRow(
        fact_id=make_fact_id(CIK, "us-gaap", concept, "USD", start, end, accession),
        cik=CIK,
        taxonomy="us-gaap",
        concept=concept,
        label=concept,
        unit="USD",
        period_start=start,
        period_end=end,
        value=Decimal(value),
        accession=accession,
        form="10-K",
        filed=filed,
        fiscal_year=None,
        fiscal_period="FY",
        frame=None,
    )


@pytest.fixture
def facts(session: Session) -> Session:
    session.add(Company(cik=CIK, name="Factco", ticker="FCTX9", sector="Test"))
    session.flush()
    y21, y22 = (date(2021, 1, 1), date(2021, 12, 31)), (date(2022, 1, 1), date(2022, 12, 31))
    session.add_all(
        [
            # FY2021 capex: reported in the FY2021 10-K, then restated in the FY2022 10-K.
            fact("Capex", 100, *y21, "A-2021", date(2022, 2, 1)),
            fact("Capex", 110, *y21, "A-2022", date(2023, 2, 1)),
            # FY2022 capex, reported once.
            fact("Capex", 130, *y22, "A-2022", date(2023, 2, 1)),
            # A quarterly value: excluded from annual values.
            fact("Capex", 30, date(2022, 1, 1), date(2022, 3, 31), "A-2022", date(2023, 2, 1)),
            # Balance-sheet instants: one at fiscal year end, one mid-year (excluded).
            fact("Assets", 900, None, date(2022, 12, 31), "A-2022", date(2023, 2, 1)),
            fact("Assets", 800, None, date(2022, 6, 30), "A-2022", date(2023, 2, 1)),
        ]
    )
    session.flush()
    return session


def test_original_value_with_restatement(facts: Session) -> None:
    [cv] = annual_values(facts, CIK, "Capex", [2021])
    [v] = cv.values
    assert v.value == 100 and v.accession == "A-2021"  # as originally reported
    assert v.restated == [
        {
            "value": "110",
            "fact_id": make_fact_id(
                CIK, "us-gaap", "Capex", "USD", date(2021, 1, 1), date(2021, 12, 31), "A-2022"
            ),
            "filed": "2023-02-01",
        }
    ]


def test_latest_years_exclude_quarters(facts: Session) -> None:
    [cv] = annual_values(facts, CIK, "Capex")
    assert [(v.fiscal_year, v.value) for v in cv.values] == [(2022, 130), (2021, 100)]
    assert cv.values[0].restated == []


def test_instants_only_at_fiscal_year_end(facts: Session) -> None:
    assert fiscal_year_ends(facts, CIK) == {date(2021, 12, 31), date(2022, 12, 31)}
    [cv] = annual_values(facts, CIK, "Assets")
    assert [(v.fiscal_year, v.value, v.period_start) for v in cv.values] == [(2022, 900, None)]


def test_aliases() -> None:
    assert (
        alias_concepts("What was FY2018 CapEx?")[0] == "PaymentsToAcquirePropertyPlantAndEquipment"
    )
    assert "Revenues" in alias_concepts("total revenue")
    assert alias_concepts("goodwill impairment") == []
