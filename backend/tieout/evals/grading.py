"""Graders: numeric tolerance for numeric answers, an LLM judge for text answers,
and a rule for whether a citation supports the answer."""

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

from tieout.agent.validation import SCALES, figures_in_answer, numbers_in
from tieout.evals.sets import EvalItem
from tieout.llm.types import ChatResponse, Message, Usage


@dataclass
class Grade:
    correct: bool
    method: str  # "numeric" | "judge" | "no_answer"
    detail: str  # matched figure, or the judge's reasoning
    judge_usage: Usage = field(default_factory=Usage)


def numeric_match(item: EvalItem, answer: str) -> tuple[bool, str]:
    """Does any figure in the answer equal the gold value (scale-aware)?

    Tolerance: `tolerance_pct` relative, or the rounding of the figure as
    written (e.g. "1.6 billion" ±0.05 billion). Lenient by design: any figure
    in the answer may match; reported as such in the README.
    """
    assert item.gold_value is not None
    gold = abs(item.gold_value)
    rel = item.tolerance_pct / 100
    for token, value in figures_in_answer(answer):
        plain = token.strip("()$%").replace(",", "")
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        rounding = Decimal("0.5") * Decimal(10) ** -decimals
        x = abs(value)
        # Either side may be the scaled one: gold "1577" (millions) vs answer
        # "1577000000", or gold "8.70" (%) vs answer "0.087".
        for scale in SCALES:
            if abs(x * scale - gold) <= max(gold * rel, rounding * scale):
                return True, f"{token} matches {item.gold_value} (x{scale})"
            if abs(x - gold * scale) <= max(gold * scale * rel, rounding):
                return True, f"{token} matches {item.gold_value} (/{scale})"
    return False, "no figure in the answer matches the gold value"


class ChatModel(Protocol):
    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> ChatResponse: ...


JUDGE_PROMPT = """\
You grade answers to questions about companies' 10-K filings.

Question: {question}

Gold answer: {gold}
Gold justification: {justification}

Answer to grade: {answer}

The answer is correct if it reaches the same conclusion as the gold answer and \
any key figures agree (small rounding differences are fine). Extra correct \
detail is fine. It is incorrect if it contradicts the gold answer, misses its \
main point, or says the information is unavailable.

Reply with only a JSON object: {{"reasoning": "<one or two sentences>", \
"verdict": "correct" or "incorrect"}}"""


def judge(llm: ChatModel, item: EvalItem, answer: str) -> Grade:
    prompt = JUDGE_PROMPT.format(
        question=item.question,
        gold=item.answer,
        justification=item.justification or "(none)",
        answer=answer,
    )
    resp = llm.chat([{"role": "user", "content": prompt}], max_tokens=400)
    text = resp.content or ""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = str(data.get("verdict", "")).lower()
    reasoning = str(data.get("reasoning") or text[:300])
    if verdict not in ("correct", "incorrect"):
        return Grade(False, "judge", f"unparseable judge reply: {text[:200]}", resp.usage)
    return Grade(verdict == "correct", "judge", reasoning, resp.usage)


def grade(item: EvalItem, answer: str | None, judge_llm: ChatModel | None) -> Grade:
    if not answer:
        return Grade(False, "no_answer", "no answer produced")
    if item.answer_kind == "numeric":
        ok, detail = numeric_match(item, answer)
        return Grade(ok, "numeric", detail)
    if judge_llm is None:
        raise ValueError("text answers need a judge model")
    return judge(judge_llm, item, answer)


DISTINCTIVE_MIN = Decimal(100)  # smaller figures ("31", "2.00") match by coincidence


def _is_year(v: Decimal) -> bool:
    return v == v.to_integral_value() and 1900 <= v <= 2100


def _cited_values(citation: dict[str, Any]) -> list[Decimal]:
    """What the citation asserts: the numbers in its quote (chunks) or its value (facts)."""
    if citation.get("kind") == "fact" or citation["source_id"].startswith("xf_"):
        return [abs(Decimal(v)) for v in citation.get("values", [])]
    return [abs(v) for v in numbers_in(citation.get("quote", ""))]


def _same_figure(cited: Decimal, evidence: Decimal) -> bool:
    """Equal at some scale, up to the rounding of the evidence figure as written
    ("7,453" in millions matches 7,452,600,000 but not 7,460,000,000)."""
    exponent = evidence.normalize().as_tuple().exponent
    step = Decimal(10) ** (exponent if isinstance(exponent, int) and exponent < 0 else 0)
    return any(abs(cited - evidence * scale) <= step * scale / 2 for scale in SCALES)


def citation_supports(item: EvalItem, citation: dict[str, Any]) -> bool:
    """Does a citation point at the gold evidence? Computed at report time.

    A citation supports the answer if it is
    - one of the gold XBRL facts (Set B), or
    - a chunk on a gold evidence page of the right filing (FinanceBench), or
    - a source asserting the gold value (numeric questions, answer tolerance), or
    - a source asserting a distinctive figure (>= 100, not a year) that also
      appears in the gold evidence passages, equal up to that figure's rounding.
    """
    if citation["source_id"] in item.gold_fact_ids:
        return True
    if (
        citation.get("kind") == "chunk"
        and citation.get("page") in item.gold_pages
        and citation.get("fiscal_year") in (None, item.fiscal_year)
    ):
        return True
    cited = _cited_values(citation)
    if item.gold_value is not None and item.gold_value != 0:
        gold = abs(item.gold_value)
        tol = gold * item.tolerance_pct / 100
        if any(abs(v - gold * s) <= tol * s for v in cited for s in SCALES):
            return True
    distinctive = [e for e in item.gold_evidence_values if e >= DISTINCTIVE_MIN and not _is_year(e)]
    return any(_same_figure(v, e) for v in cited for e in distinctive)
