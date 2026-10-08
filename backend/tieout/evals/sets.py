"""Eval sets.

Set A (financebench): the FinanceBench open-source questions about the
supported 10-Ks, with gold answers, justifications and evidence pages.
FinanceBench is by Patronus AI, CC BY-NC 4.0; used for non-commercial evaluation.

Set B (xbrl_numeric): numeric questions generated from XBRL with exact
Decimal ground truth: single-value lookups and year-over-year % changes.
Ground truth comes from XBRL, so this set favours the XBRL tool by design.
"""

import json
import random
import re
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from tieout.edgar.facts_query import annual_values, fiscal_year_ends
from tieout.financebench.dataset import Document, Question
from tieout.financebench.selection import CompanyConfig

AnswerKind = Literal["numeric", "text"]


class EvalItem(BaseModel):
    id: str
    eval_set: Literal["financebench", "xbrl_numeric"]
    question: str
    answer: str  # gold answer as text
    answer_kind: AnswerKind
    gold_value: Decimal | None = None  # for numeric answers
    tolerance_pct: Decimal = Decimal("1")  # relative tolerance for numeric grading
    ticker: str
    fiscal_year: int
    doc_name: str
    justification: str | None = None
    gold_pages: list[int] = Field(default_factory=list)  # 1-based PDF pages
    gold_fact_ids: list[str] = Field(default_factory=list)
    question_type: str = ""


# --- Set A: FinanceBench ----------------------------------------------------------

NUMERIC_ANSWER_RE = re.compile(r"^\s*\(?-?\$?\s*-?[\d,]*\.?\d+\s*%?\)?\s*$")


def parse_numeric_answer(answer: str) -> Decimal | None:
    """'$1577.00' -> 1577.00, '8.70%' -> 8.70, '0.66' -> 0.66; prose -> None."""
    text = answer.strip().rstrip(".")
    for unit in (" million", " billion", " USD", " usd"):
        text = text.removesuffix(unit)
    if not NUMERIC_ANSWER_RE.match(text):
        return None
    negative = text.strip().startswith(("(", "-")) or "-" in text
    digits = re.sub(r"[^\d.]", "", text)
    if not digits or digits == ".":
        return None
    value = Decimal(digits)
    return -value if negative else value


def build_financebench_set(
    questions: list[Question], docs: dict[str, Document], companies: list[CompanyConfig]
) -> list[EvalItem]:
    filings = {f.doc_name: (c, f) for c in companies for f in c.filings}
    items = []
    for q in questions:
        if q.doc_name not in filings or docs[q.doc_name].doc_type != "10k":
            continue
        company, filing = filings[q.doc_name]
        value = parse_numeric_answer(q.answer)
        items.append(
            EvalItem(
                id=q.financebench_id,
                eval_set="financebench",
                question=q.question,
                answer=q.answer,
                answer_kind="numeric" if value is not None else "text",
                gold_value=value,
                ticker=company.ticker,
                fiscal_year=filing.fiscal_year,
                doc_name=q.doc_name,
                justification=q.justification,
                gold_pages=sorted(
                    {e.evidence_page_num + 1 for e in q.evidence if e.doc_name == q.doc_name}
                ),
                question_type=q.question_type,
            )
        )
    return sorted(items, key=lambda i: i.id)


# --- Set B: XBRL numeric --------------------------------------------------------------

# (friendly name, unit word, concepts in preference order). Flows and balances
# that every 10-K reports, so questions are answerable from any filing.
METRICS: list[tuple[str, str, list[str]]] = [
    (
        "total revenue",
        "USD millions",
        [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "RevenuesNetOfInterestExpense",
        ],
    ),
    ("net income", "USD millions", ["NetIncomeLoss"]),
    ("operating income", "USD millions", ["OperatingIncomeLoss"]),
    (
        "capital expenditure (purchases of property, plant and equipment)",
        "USD millions",
        ["PaymentsToAcquirePropertyPlantAndEquipment"],
    ),
    (
        "net cash provided by operating activities",
        "USD millions",
        ["NetCashProvidedByUsedInOperatingActivities"],
    ),
    ("total assets", "USD millions", ["Assets"]),
    ("research and development expense", "USD millions", ["ResearchAndDevelopmentExpense"]),
]
MILLION = Decimal(1_000_000)


def build_xbrl_set(
    session: Session,
    companies: list[CompanyConfig],
    n_lookup: int = 20,
    n_growth: int = 20,
    seed: int = 7,
) -> list[EvalItem]:
    """Deterministic sample of lookup and YoY-growth questions per filing."""
    lookups: list[EvalItem] = []
    growth: list[EvalItem] = []
    for company in companies:
        year_ends = fiscal_year_ends(session, company.cik)
        for filing in company.filings:
            fy = filing.fiscal_year
            for name, unit, concepts in METRICS:
                for concept in concepts:
                    found = annual_values(session, company.cik, concept, [fy, fy - 1], year_ends)
                    usd = [cv for cv in found if cv.unit == "USD"]
                    if not usd:
                        continue
                    by_year = {v.fiscal_year: v for v in usd[0].values}
                    cur = by_year.get(fy)
                    if cur is None:
                        continue
                    lookups.append(
                        EvalItem(
                            id=f"xbrl-{company.ticker}-{fy}-{concept}-lookup",
                            eval_set="xbrl_numeric",
                            question=(
                                f"What was {company.name}'s {name} in fiscal year {fy}? "
                                f"Answer in {unit}."
                            ),
                            answer=f"{cur.value / MILLION:f}",
                            answer_kind="numeric",
                            gold_value=cur.value / MILLION,
                            tolerance_pct=Decimal("0.5"),
                            ticker=company.ticker,
                            fiscal_year=fy,
                            doc_name=filing.doc_name,
                            gold_fact_ids=[cur.fact_id],
                            question_type="lookup",
                        )
                    )
                    prev = by_year.get(fy - 1)
                    # Only when the prior year was never restated: one unambiguous answer.
                    if prev is not None and not prev.restated and prev.value != 0:
                        pct = (cur.value - prev.value) / abs(prev.value) * 100
                        growth.append(
                            EvalItem(
                                id=f"xbrl-{company.ticker}-{fy}-{concept}-growth",
                                eval_set="xbrl_numeric",
                                question=(
                                    f"By what percentage did {company.name}'s {name} change "
                                    f"from fiscal year {fy - 1} to fiscal year {fy}? "
                                    "Give the percentage rounded to one decimal place."
                                ),
                                answer=f"{pct:.1f}%",
                                answer_kind="numeric",
                                gold_value=pct.quantize(Decimal("0.0001")),
                                tolerance_pct=Decimal("1"),
                                ticker=company.ticker,
                                fiscal_year=fy,
                                doc_name=filing.doc_name,
                                gold_fact_ids=[cur.fact_id, prev.fact_id],
                                question_type="yoy_growth",
                            )
                        )
                    break  # first concept the company reports wins
    rng = random.Random(seed)
    picked = stratified(lookups, n_lookup, rng) + stratified(growth, n_growth, rng)
    return sorted(picked, key=lambda i: i.id)


def stratified(items: list[EvalItem], n: int, rng: random.Random) -> list[EvalItem]:
    """Round-robin across companies, so companies with more filings aren't over-weighted."""
    by_ticker: dict[str, list[EvalItem]] = {}
    for item in sorted(items, key=lambda i: i.id):
        by_ticker.setdefault(item.ticker, []).append(item)
    for group in by_ticker.values():
        rng.shuffle(group)
    order = sorted(by_ticker)
    picked: list[EvalItem] = []
    while len(picked) < n and any(by_ticker.values()):
        for ticker in order:
            if by_ticker[ticker] and len(picked) < n:
                picked.append(by_ticker[ticker].pop())
    return picked


def write_set(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(i.model_dump(mode="json"), ensure_ascii=False) for i in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_set(path: Path) -> list[EvalItem]:
    return [
        EvalItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
