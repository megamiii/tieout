import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tieout.agent.tools import Tool, chunk_evidence
from tieout.agent.validation import Evidence
from tieout.evals import metrics, runner
from tieout.evals.baseline import parse_json_answer, question_filters, run_baseline
from tieout.evals.grading import citation_supports, grade, judge, numeric_match
from tieout.evals.sets import EvalItem, parse_numeric_answer, stratified
from tieout.financebench.selection import CompanyConfig, FilingConfig
from tieout.llm.limits import UsageLedger
from tieout.llm.types import ChatResponse, Message, Usage


def item(**kw: Any) -> EvalItem:
    base: dict[str, Any] = {
        "id": "q1",
        "eval_set": "financebench",
        "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
        "answer": "$1577.00",
        "answer_kind": "numeric",
        "gold_value": Decimal("1577.00"),
        "ticker": "MMM",
        "fiscal_year": 2018,
        "doc_name": "3M_2018_10K",
        "gold_pages": [60],
    }
    return EvalItem(**{**base, **kw})


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("$1577.00", Decimal("1577.00")),
        ("8.70%", Decimal("8.70")),
        ("0.66", Decimal("0.66")),
        ("-3.7", Decimal("-3.7")),
        ("Yes. Corning had a positive working capital of $831 million", None),
        ("The quick ratio was 0.54", None),
    ],
)
def test_parse_numeric_answer(text: str, value: Decimal | None) -> None:
    assert parse_numeric_answer(text) == value


@pytest.mark.parametrize(
    ("answer", "ok"),
    [
        ("3M's FY2018 capex was $1,577 million.", True),
        ("Capex was $1.58 billion in FY2018.", True),  # rounding of 1.58 bn covers 1.577
        ("Capex was 1577000000 USD.", True),
        ("Capex was $1,373 million.", False),
        ("Capex was about $1.6 billion.", True),  # ±0.05 bn rounding
        ("Capex was about $1.9 billion.", False),
    ],
)
def test_numeric_match(answer: str, ok: bool) -> None:
    assert numeric_match(item(), answer)[0] is ok


def test_percent_answers_match_ratios() -> None:
    it = item(answer="8.70%", gold_value=Decimal("8.70"))
    assert numeric_match(it, "The margin was 8.7%.")[0]
    assert numeric_match(it, "The ratio is 0.087.")[0]
    assert not numeric_match(it, "The margin was 9.4%.")[0]


class Scripted:
    model = "judge"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[list[Message]] = []

    def chat(self, messages: list[Message], **kw: Any) -> ChatResponse:
        self.prompts.append(messages)
        return ChatResponse(
            self.replies.pop(0), [], "stop", Usage(50, 10), "judge", "h", False, 1.0
        )


def test_judge_parses_verdict_and_logs_reasoning() -> None:
    it = item(
        answer="Yes, it has positive working capital.",
        answer_kind="text",
        gold_value=None,
        justification="Current assets exceed current liabilities.",
    )
    llm = Scripted(['```json\n{"reasoning": "Same conclusion.", "verdict": "correct"}\n```'])
    g = grade(it, "Yes: working capital was positive.", llm)
    assert g.correct and g.method == "judge" and g.detail == "Same conclusion."
    prompt = llm.prompts[0][0]["content"]
    assert "Current assets exceed" in prompt and "working capital was positive" in prompt


def test_unparseable_judge_reply_counts_as_incorrect() -> None:
    it = item(answer_kind="text", gold_value=None)
    g = judge(Scripted(["I think it's fine"]), it, "answer")
    assert not g.correct and "unparseable" in g.detail


def test_no_answer_is_incorrect_without_calling_judge() -> None:
    g = grade(item(answer_kind="text", gold_value=None), None, None)
    assert not g.correct and g.method == "no_answer"


def test_citation_support_rules() -> None:
    it = item(
        gold_fact_ids=["xf_aaaaaaaaaaaaaaaa"],
        answer_kind="text",
        gold_value=None,
        gold_evidence_values=[Decimal(7453), Decimal(5175)],
    )
    on_gold_page = {"source_id": "ch_1", "kind": "chunk", "page": 60, "fiscal_year": 2018}
    wrong_filing = {"source_id": "ch_1", "kind": "chunk", "page": 60, "fiscal_year": 2022}
    other_page = {"source_id": "ch_2", "kind": "chunk", "page": 12, "quote": "5 segments in 2018"}
    # A fact can't be "on a page"; it supports if it states a gold-evidence figure.
    fact = {"source_id": "xf_bbbbbbbbbbbbbbbb", "kind": "fact", "values": ["7453000000"]}
    unrelated_fact = {"source_id": "xf_cccccccccccccccc", "kind": "fact", "values": ["999"]}
    assert citation_supports(it, on_gold_page)
    assert not citation_supports(it, wrong_filing)
    assert not citation_supports(it, other_page)
    assert citation_supports(it, fact)
    assert not citation_supports(it, unrelated_fact)
    assert citation_supports(it, {"source_id": "xf_aaaaaaaaaaaaaaaa"})  # gold fact id
    numeric = item()  # gold value 1577 (USD millions)
    assert citation_supports(numeric, {"source_id": "ch_3", "kind": "chunk", "quote": "(1,577)"})


def test_citation_support_ignores_coincidental_figures() -> None:
    it = item(
        answer_kind="text",
        gold_value=None,
        gold_evidence_values=[Decimal(31), Decimal(2019), Decimal(9912)],
    )
    small = {"source_id": "ch_1", "kind": "chunk", "page": 5, "quote": "31 aircraft"}
    year = {"source_id": "ch_2", "kind": "chunk", "page": 5, "quote": "in 2019"}
    near_miss = {"source_id": "xf_dddddddddddddddd", "kind": "fact", "values": ["9961000000"]}
    exact = {"source_id": "xf_eeeeeeeeeeeeeeee", "kind": "fact", "values": ["9912000000"]}
    assert not citation_supports(it, small)  # figures under 100 match by coincidence
    assert not citation_supports(it, year)
    assert not citation_supports(it, near_miss)  # 0.5% off is a different number
    assert citation_supports(it, exact)


def company(ticker: str, name: str, years: list[int]) -> CompanyConfig:
    filings = [
        FilingConfig(
            doc_name=f"{ticker}_{y}",
            fiscal_year=y,
            accession=f"a{y}",
            form="10-K",
            period_end=f"{y}-12-31",
            filing_date=f"{y + 1}-02-01",
            edgar_url="u",
            pdf_url="u",
            question_count=1,
        )
        for y in years
    ]
    return CompanyConfig(
        name=name, ticker=ticker, cik=1, sector="s", question_count=1, filings=filings
    )


COMPANIES = [company("MMM", "3M", [2018, 2022]), company("AMD", "AMD", [2015, 2022])]


def test_question_filters_use_only_string_matching() -> None:
    c, y = question_filters("What is the FY2018 capex for 3M?", COMPANIES)
    assert c is not None and c.ticker == "MMM" and y == 2018
    # FY2021 is not a filing we have; FY2022 is the filing that reports it.
    c, y = question_filters("Did AMD's FY2021 to FY2022 revenue grow?", COMPANIES)
    assert c is not None and c.ticker == "AMD" and y == 2022
    assert question_filters("Who is the CEO of Boeing?", COMPANIES) == (None, None)


class StubSearch(Tool):
    from tieout.agent.tools import SearchFilingsArgs as args_model

    name = "search_filings"
    description = "x"

    def run(self, args: Any) -> dict[str, Any]:
        assert args.ticker == "MMM" and args.fiscal_year == 2018
        return {
            "results": [
                {
                    "chunk_id": "ch_be9a964b06a211ba",
                    "ticker": "MMM",
                    "fiscal_year": 2018,
                    "section": "Item 8",
                    "page": 60,
                    "text": "Purchases of PP&E (1,577)",
                }
            ]
        }

    def evidence(self, output: Any) -> list[Evidence]:
        return chunk_evidence(output["results"])


def test_baseline_single_call_with_json_answer() -> None:
    reply = json.dumps(
        {
            "answer": "$1,577 million",
            "confidence": 0.9,
            "needs_review": False,
            "citations": [{"source_id": "ch_be9a964b06a211ba", "quote": "(1,577)"}],
        }
    )
    llm = Scripted([reply])
    final, responses, _, _ = run_baseline(
        llm, StubSearch(), COMPANIES, "What is the FY2018 capex for 3M?"
    )
    assert final.status == "answered" and final.answer == "$1,577 million"
    assert final.citations[0]["page"] == 60 and len(responses) == 1
    assert "[ch_be9a964b06a211ba]" in llm.prompts[0][0]["content"]


def test_baseline_invalid_json_needs_review() -> None:
    final, *_ = run_baseline(
        Scripted(["The answer is 1577."]),
        StubSearch(),
        COMPANIES,
        "What is the FY2018 capex for 3M?",
    )
    assert final.status == "needs_review"
    assert parse_json_answer('noise {"answer": "x"} noise') == {"answer": "x"}


def test_stratified_sampling_balances_companies() -> None:
    import random

    items = [item(id=f"{t}-{i}", ticker=t) for t in ("A", "B") for i in range(10)]
    items += [item(id="C-0", ticker="C")]
    picked = stratified(items, 5, random.Random(0))
    tickers = [i.ticker for i in picked]
    assert tickers.count("C") == 1 and abs(tickers.count("A") - tickers.count("B")) <= 1


def test_budget_stops_before_the_daily_limit(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path, today=lambda: date(2026, 10, 9))
    budget = runner.Budget(ledger, "m", daily_limit=20, reserve=8)
    assert not budget.exhausted()
    for _ in range(13):
        ledger.add("m", Usage(1, 1), cached=False)
    assert budget.exhausted()  # 13 used + 8 reserved > 20
    assert not runner.Budget(ledger, "m", None, 8).exhausted()  # unknown limit


def test_metrics_from_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    items = [item(id=f"q{i}") for i in range(4)]

    def save(i: int, status: str, correct: bool, supports: list[bool], latency: float) -> None:
        runner.save_result(
            "agent_full",
            f"q{i}",
            {
                "status": status,
                "requests": 4,
                "prompt_tokens": 900,
                "completion_tokens": 100,
                "latency_ms": latency,
                "citations": [
                    {"source_id": f"ch_{n}", "kind": "chunk", "quote": "1,577" if s else "1"}
                    for n, s in enumerate(supports)
                ],
                "grade": {"correct": correct},
            },
        )

    save(0, "answered", True, [True, True], 1000)
    save(1, "answered", False, [False], 2000)
    save(2, "needs_review", False, [], 3000)
    save(3, "needs_review", True, [True], 4000)
    m = metrics.compute("agent_full", "financebench", items)
    assert m.accuracy == 0.5 and m.answered_rate == 0.5 and m.accuracy_answered == 0.5
    assert m.citation_precision == 0.75 and m.citations == 4
    assert m.abstention_precision == 0.5  # 1 of 2 reviewed answers was wrong
    assert m.abstention_recall == 0.5  # 1 of 2 wrong answers was sent to review
    assert m.requests_per_item == 4 and m.tokens_per_item == 1000
    assert m.latency_p50_s in (2.0, 3.0) and m.latency_p95_s == 4.0
