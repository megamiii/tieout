"""Experiment 1 baseline: a single prompt over vector-retrieved chunks.

No agent loop and no tools for the model: retrieve the top-k chunks by vector
similarity (company and fiscal year filters taken from the question by plain
string matching), put them in one prompt, and ask for a JSON answer with
citations. Same output shape as the agent, so the same graders apply.
"""

import json
import re
import time
from typing import Any

from tieout.agent.loop import FinalAnswer
from tieout.agent.tools import SearchFilingsArgs, Tool
from tieout.agent.validation import EvidenceLedger
from tieout.evals.grading import ChatModel
from tieout.financebench.selection import CompanyConfig
from tieout.llm.types import ChatResponse

TOP_K = 8
CONFIDENCE_THRESHOLD = 0.7

# Names a question may use for each company (FinanceBench uses the full name).
NAME_VARIANTS = {
    "MMM": ["3M"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "BA": ["Boeing"],
    "AXP": ["American Express", "AmEx"],
    "PEP": ["PepsiCo", "Pepsi"],
    "ADBE": ["Adobe"],
    "AMCR": ["Amcor"],
    "BBY": ["Best Buy"],
    "VZ": ["Verizon"],
    "GLW": ["Corning"],
    "CVS": ["CVS"],
    "GIS": ["General Mills"],
    "NKE": ["Nike"],
}

BASELINE_PROMPT = """\
Answer the question about a company's 10-K using only the passages below.

Question: {question}

Passages:
{passages}

Reply with only a JSON object:
{{"answer": "<concise answer with units and fiscal year>",
  "confidence": <probability 0-1 that the answer is correct>,
  "citations": [{{"source_id": "<chunk_id>", "quote": "<exact text from that passage>"}}],
  "needs_review": <true if the passages do not contain the answer>}}"""


def question_filters(
    question: str, companies: list[CompanyConfig]
) -> tuple[CompanyConfig | None, int | None]:
    """Company and fiscal year from the question text, by string matching only."""
    company = None
    for c in companies:
        names = NAME_VARIANTS.get(c.ticker, [c.name])
        if any(re.search(rf"\b{re.escape(n)}\b", question, re.IGNORECASE) for n in names):
            company = c
            break
    years = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", question)]
    year = None
    if company is not None:
        available = {f.fiscal_year for f in company.filings}
        candidates = [y for y in years if y in available]
        year = max(candidates) if candidates else None
    return company, year


def parse_json_answer(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def run_baseline(
    llm: ChatModel,
    search_tool: Tool,
    companies: list[CompanyConfig],
    question: str,
) -> tuple[FinalAnswer, list[ChatResponse], EvidenceLedger, float]:
    """Returns the final answer, LLM responses, evidence, and tool latency (ms)."""
    company, year = question_filters(question, companies)
    args = SearchFilingsArgs(
        query=question, ticker=company.ticker if company else None, fiscal_year=year, k=TOP_K
    )
    start = time.perf_counter()
    output = search_tool.run(args)
    tool_ms = (time.perf_counter() - start) * 1000
    ledger = EvidenceLedger()
    for ev in search_tool.evidence(output):
        ledger.add(ev)
    passages = "\n\n".join(
        f"[{r['chunk_id']}] ({r['ticker']} FY{r['fiscal_year']}, {r['section']}, p.{r['page']})\n"
        f"{r['text']}"
        for r in output.get("results", [])
    )
    resp = llm.chat(
        [{"role": "user", "content": BASELINE_PROMPT.format(question=question, passages=passages)}],
        max_tokens=1024,
    )
    data = parse_json_answer(resp.content or "")
    if data is None:
        final = FinalAnswer(
            status="needs_review",
            answer=resp.content,
            confidence=None,
            rationale=None,
            citations=[],
            review_reasons=["reply was not valid JSON"],
        )
        return final, [resp], ledger, tool_ms

    citations = []
    for c in data.get("citations") or []:
        if isinstance(c, dict) and "source_id" in c:
            source = ledger.sources.get(str(c["source_id"]))
            citations.append(
                {
                    "source_id": str(c["source_id"]),
                    "quote": str(c.get("quote", "")),
                    "kind": source.kind if source else "unknown",
                    **(source.meta if source else {}),
                }
            )
    try:
        confidence = float(data.get("confidence"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        confidence = 0.0
    reasons = []
    if data.get("needs_review"):
        reasons.append("model flagged for review")
    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"confidence {confidence:.2f} is below {CONFIDENCE_THRESHOLD:.2f}")
    final = FinalAnswer(
        status="needs_review" if reasons else "answered",
        answer=str(data.get("answer") or ""),
        confidence=confidence,
        rationale=None,
        citations=citations,
        review_reasons=reasons,
    )
    return final, [resp], ledger, tool_ms
