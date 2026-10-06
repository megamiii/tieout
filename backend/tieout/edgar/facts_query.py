"""Look up XBRL facts the way an auditor would read them.

- Concept lookup: an exact concept name, or a phrase matched to the
  company's concepts by curated aliases, embedding similarity over
  label + description, and keyword matches on the label.
- Annual values: one value per fiscal year, *as originally reported* (the
  earliest-filed 10-K that reports that period). FinanceBench answers use the
  figures in a specific 10-K, and an auditor ties out to the original filing.
- Restatements: if a later 10-K reports a different value for the same
  period, it is returned too. Conflicting sources are a reason to route an
  answer to human review.
"""

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import case, func, literal, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from tieout.db.models import XbrlConcept, XbrlFactRow
from tieout.edgar.filings import fiscal_year_label
from tieout.retrieval.embeddings import Embedder, Vector
from tieout.retrieval.search import rrf

ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 350, 380
DESCRIPTION_CHARS = 300

# Common metric names -> us-gaap concepts. Filers choose between equivalent
# concepts (e.g. three revenue tags), so several are listed in preference order.
ALIASES: dict[str, list[str]] = {
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "capital expenditure": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net sales": ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net income": ["NetIncomeLoss", "ProfitLoss"],
    "operating income": ["OperatingIncomeLoss"],
    "gross profit": ["GrossProfit"],
    "cost of goods sold": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "cost of revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "total assets": ["Assets"],
    "total liabilities": ["Liabilities"],
    "current assets": ["AssetsCurrent"],
    "current liabilities": ["LiabilitiesCurrent"],
    "inventory": ["InventoryNet"],
    "accounts receivable": ["AccountsReceivableNetCurrent"],
    "cash and cash equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash from operations": ["NetCashProvidedByUsedInOperatingActivities"],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "dividends paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "long-term debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "shareholders equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "diluted eps": ["EarningsPerShareDiluted"],
    "basic eps": ["EarningsPerShareBasic"],
    "research and development": ["ResearchAndDevelopmentExpense"],
    "income tax": ["IncomeTaxExpenseBenefit"],
    "interest expense": ["InterestExpense"],
}


def alias_concepts(query: str) -> list[str]:
    q = " ".join(re.findall(r"[a-z]+", query.lower()))
    out: list[str] = []
    for phrase, concepts in ALIASES.items():
        if phrase.replace("-", " ") in q:
            out += [c for c in concepts if c not in out]
    return out


def concept_text(label: str, description: str) -> str:
    return f"{label}. {description[:DESCRIPTION_CHARS]}"


def build_concept_catalog(
    session: Session,
    companyfacts_paths: list[Path],
    embed_texts: Callable[[list[str]], list[Vector]],
) -> int:
    """(Re)build xbrl_concepts from companyfacts files; returns the concept count."""
    catalog: dict[str, tuple[str, str, str]] = {}
    for path in companyfacts_paths:
        facts = json.loads(path.read_text(encoding="utf-8"))["facts"]
        for taxonomy in ("us-gaap", "dei"):
            for concept, body in facts.get(taxonomy, {}).items():
                if concept not in catalog:
                    catalog[concept] = (
                        taxonomy,
                        body.get("label") or concept,
                        body.get("description") or "",
                    )
    names = sorted(catalog)
    vectors = embed_texts([concept_text(catalog[n][1], catalog[n][2]) for n in names])
    for name, vec in zip(names, vectors, strict=True):
        taxonomy, label, description = catalog[name]
        values = {
            "taxonomy": taxonomy,
            "label": label,
            "description": description,
            "embedding": [float(x) for x in vec],
        }
        session.execute(
            pg_insert(XbrlConcept)
            .values(concept=name, **values)
            .on_conflict_do_update(index_elements=[XbrlConcept.concept], set_=values)
        )
    return len(names)


def find_concepts(
    session: Session, embedder: Embedder, cik: int, query: str, k: int = 3
) -> list[str]:
    """Best-matching concepts that this company actually reports in 10-Ks."""
    reported = select(XbrlFactRow.concept).where(XbrlFactRow.cik == cik).distinct()
    available = set(session.scalars(reported))
    aliases = [c for c in alias_concepts(query) if c in available]

    qvec = embedder.embed_query(query).tolist()
    vector = list(
        session.scalars(
            select(XbrlConcept.concept)
            .where(XbrlConcept.concept.in_(reported))
            .order_by(XbrlConcept.embedding.cosine_distance(qvec))
            .limit(20)
        )
    )
    words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2]
    keyword: list[str] = []
    if words:
        label = func.lower(XbrlConcept.label)
        matched = sum(
            (case((func.strpos(label, w) > 0, 1), else_=0) for w in words), start=literal(0)
        )
        keyword = list(
            session.scalars(
                select(XbrlConcept.concept)
                .where(XbrlConcept.concept.in_(reported), matched > 0)
                .order_by(matched.desc(), func.length(XbrlConcept.label))
                .limit(20)
            )
        )
    fused = [c for c, _ in rrf([vector, keyword]) if c not in aliases]
    return (aliases + fused)[:k]


@dataclass
class ReportedValue:
    fiscal_year: int
    period_start: date | None
    period_end: date
    value: Decimal
    fact_id: str
    accession: str
    filed: date
    restated: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ConceptValues:
    concept: str
    label: str
    unit: str
    values: list[ReportedValue]


def fiscal_year_ends(session: Session, cik: int) -> set[date]:
    """Period-end dates of the company's annual (~1 year) duration facts."""
    days = XbrlFactRow.period_end - XbrlFactRow.period_start
    rows = session.scalars(
        select(XbrlFactRow.period_end)
        .where(
            XbrlFactRow.cik == cik,
            XbrlFactRow.period_start.is_not(None),
            days.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS),
        )
        .distinct()
    )
    return set(rows)


def annual_values(
    session: Session,
    cik: int,
    concept: str,
    fiscal_years: list[int] | None = None,
    year_ends: set[date] | None = None,
    latest: int = 3,
) -> list[ConceptValues]:
    """One value per (unit, fiscal year): the original report, with restatements."""
    rows = list(
        session.scalars(
            select(XbrlFactRow).where(XbrlFactRow.cik == cik, XbrlFactRow.concept == concept)
        )
    )
    year_ends = year_ends if year_ends is not None else fiscal_year_ends(session, cik)
    groups: dict[tuple[str, date | None, date], list[XbrlFactRow]] = defaultdict(list)
    for r in rows:
        if r.period_start is not None:
            if not ANNUAL_MIN_DAYS <= (r.period_end - r.period_start).days <= ANNUAL_MAX_DAYS:
                continue  # quarterly or multi-year durations
        elif r.period_end not in year_ends:
            continue  # instants that are not fiscal year-end balances
        groups[(r.unit, r.period_start, r.period_end)].append(r)

    by_unit: dict[str, list[ReportedValue]] = defaultdict(list)
    for (unit, start, end), facts in groups.items():
        facts.sort(key=lambda f: (f.filed, f.accession))
        original = facts[0]
        restated = [
            {"value": str(f.value), "fact_id": f.fact_id, "filed": f.filed.isoformat()}
            for f in facts[1:]
            if f.value != original.value
        ]
        by_unit[unit].append(
            ReportedValue(
                fiscal_year=fiscal_year_label(end),
                period_start=start,
                period_end=end,
                value=original.value,
                fact_id=original.fact_id,
                accession=original.accession,
                filed=original.filed,
                restated=restated,
            )
        )
    label = rows[0].label if rows else concept
    out: list[ConceptValues] = []
    for unit, values in by_unit.items():
        values.sort(key=lambda v: v.period_end, reverse=True)
        if fiscal_years:
            values = [v for v in values if v.fiscal_year in fiscal_years]
        else:
            values = values[:latest]
        if values:
            out.append(ConceptValues(concept=concept, label=label, unit=unit, values=values))
    return out
