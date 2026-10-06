"""Flatten SEC XBRL "company facts" JSON into one row per reported value.

companyfacts JSON shape:
  facts -> taxonomy ("us-gaap", "dei", ...) -> concept ("Revenues", ...)
        -> {label, description, units: {"USD": [ {start?, end, val, accn,
            fy, fp, form, filed, frame?}, ... ]}}

A value with `start` is a duration (e.g. revenue for a fiscal year); a value
without `start` is an instant (e.g. cash at year end).
"""

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from tieout.edgar.client import ANNUAL_FORMS


@dataclass(frozen=True)
class XbrlFact:
    fact_id: str
    cik: int
    taxonomy: str
    concept: str
    label: str
    unit: str
    period_start: date | None
    period_end: date
    value: Decimal
    accession: str
    form: str
    filed: date
    fiscal_year: int | None  # as reported by the SEC; unreliable, kept for reference
    fiscal_period: str | None
    frame: str | None

    @property
    def duration_days(self) -> int | None:
        if self.period_start is None:
            return None
        return (self.period_end - self.period_start).days


def make_fact_id(
    cik: int,
    taxonomy: str,
    concept: str,
    unit: str,
    start: date | None,
    end: date,
    accession: str,
) -> str:
    """Stable ID: the same fact gets the same ID across database rebuilds.

    Recorded demo runs and eval reports cite facts by ID, so IDs must not
    depend on insertion order the way autoincrement keys do.
    """
    key = f"{cik}|{taxonomy}|{concept}|{unit}|{start or ''}|{end}|{accession}"
    return "xf_" + hashlib.sha1(key.encode()).hexdigest()[:16]


def iter_facts(
    companyfacts: dict[str, Any],
    *,
    forms: Iterable[str] = ANNUAL_FORMS,
    taxonomies: Iterable[str] = ("us-gaap", "dei"),
) -> Iterator[XbrlFact]:
    """Yield facts from annual filings, de-duplicated by fact ID.

    The same value can appear more than once in companyfacts (e.g. once per
    `frame`); de-duplicating by ID keeps the first occurrence.
    """
    cik = int(companyfacts["cik"])
    wanted_forms = set(forms)
    seen: set[str] = set()
    for taxonomy in taxonomies:
        for concept, body in companyfacts.get("facts", {}).get(taxonomy, {}).items():
            label = body.get("label") or concept
            for unit, values in body.get("units", {}).items():
                for v in values:
                    if v.get("form") not in wanted_forms:
                        continue
                    start = date.fromisoformat(v["start"]) if v.get("start") else None
                    end = date.fromisoformat(v["end"])
                    fact_id = make_fact_id(cik, taxonomy, concept, unit, start, end, v["accn"])
                    if fact_id in seen:
                        continue
                    seen.add(fact_id)
                    yield XbrlFact(
                        fact_id=fact_id,
                        cik=cik,
                        taxonomy=taxonomy,
                        concept=concept,
                        label=label,
                        unit=unit,
                        period_start=start,
                        period_end=end,
                        value=Decimal(str(v["val"])),
                        accession=v["accn"],
                        form=v["form"],
                        filed=date.fromisoformat(v["filed"]),
                        fiscal_year=v.get("fy"),
                        fiscal_period=v.get("fp"),
                        frame=v.get("frame"),
                    )
