"""The agent loop, written by hand (no framework).

    messages = [system, user]
    repeat up to max_steps:
        response = LLM(messages, tool schemas)
        append the assistant message
        for each tool call:
            submit_answer -> validate against this run's evidence:
                valid    -> final answer (answered or needs_review), stop
                invalid  -> return the problems to the model so it can fix them
            other tools  -> run, record the evidence it showed, append the result
        a plain-text reply is not an answer: ask the model to call submit_answer
        stop if the run has used its token budget

On the last allowed step, tool_choice forces submit_answer, so a run that is
out of steps still ends with an answer that went through validation. A run
that ends without a valid submission is routed to human review.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from tieout.agent.tools import Tool, ToolOutcome, execute_tool_call
from tieout.agent.validation import EvidenceLedger, SubmitAnswerArgs, validate_submission
from tieout.llm.types import ChatResponse, Message, Usage

SUBMIT = "submit_answer"
FORCE_SUBMIT = {"type": "function", "function": {"name": SUBMIT}}
NUDGE = (
    "Do not answer in plain text. Call submit_answer with your answer, confidence, "
    "rationale and citations."
)

Status = Literal["answered", "needs_review"]
StopReason = Literal["submitted", "validation_failed", "max_steps", "token_budget"]


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


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 6
    max_tokens_per_run: int = 30_000
    max_tokens_per_response: int = 1024
    confidence_threshold: float = 0.7
    max_rejections: int = 2  # invalid submissions before giving up


DEFAULT_LIMITS = AgentLimits()


@dataclass
class FinalAnswer:
    status: Status
    answer: str | None
    confidence: float | None
    rationale: str | None
    citations: list[dict[str, Any]]
    review_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AgentStep:
    index: int
    response: ChatResponse
    tool_outcomes: list[ToolOutcome] = field(default_factory=list)
    forced_submit: bool = False
    nudged: bool = False  # the model replied in plain text and was asked to submit


@dataclass
class AgentResult:
    question: str
    final: FinalAnswer
    stop_reason: StopReason
    steps: list[AgentStep]
    usage: Usage
    messages: list[Message]
    rejections: int = 0
    evidence: EvidenceLedger = field(default_factory=EvidenceLedger)

    @property
    def answer(self) -> str | None:
        return self.final.answer

    @property
    def tool_calls(self) -> int:
        return sum(len(s.tool_outcomes) for s in self.steps)

    @property
    def cached(self) -> bool:
        """True if every LLM response came from the cache (no tokens spent)."""
        return all(s.response.cached for s in self.steps)


def finalize(args: SubmitAnswerArgs, ledger: EvidenceLedger, limits: AgentLimits) -> FinalAnswer:
    reasons = []
    if args.needs_review:
        reasons.append(f"model flagged for review: {args.review_reason or 'no reason given'}")
    if args.confidence < limits.confidence_threshold:
        reasons.append(
            f"confidence {args.confidence:.2f} is below {limits.confidence_threshold:.2f}"
        )
    citations, warnings = [], []
    for c in args.citations:
        ev = ledger.sources[c.source_id]
        citations.append({"source_id": c.source_id, "quote": c.quote, "kind": ev.kind, **ev.meta})
        for r in ev.meta.get("restated", []):
            warnings.append(
                f"{c.source_id} (FY{ev.meta.get('fiscal_year')}) was later reported as "
                f"{r['value']} in a 10-K filed {r['filed']}"
            )
    return FinalAnswer(
        status="needs_review" if reasons else "answered",
        answer=args.answer,
        confidence=args.confidence,
        rationale=args.rationale,
        citations=citations,
        review_reasons=reasons,
        warnings=warnings,
    )


def run_agent(
    llm: ChatModel,
    tools: list[Tool],
    system_prompt: str,
    question: str,
    limits: AgentLimits = DEFAULT_LIMITS,
    extra: dict[str, Any] | None = None,
) -> AgentResult:
    registry = {t.name: t for t in tools}
    if SUBMIT not in registry:
        raise ValueError("the agent needs a submit_answer tool")
    schemas = [t.schema() for t in tools]
    # "Never let the LLM do math" is enforceable only if it has a calculator.
    check_figures = "calculate" in registry
    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    ledger = EvidenceLedger()
    steps: list[AgentStep] = []
    usage = Usage()
    final: FinalAnswer | None = None
    stop: StopReason = "max_steps"
    rejections = 0

    for i in range(limits.max_steps):
        last = i == limits.max_steps - 1
        response = llm.chat(
            messages,
            tools=schemas,
            tool_choice=FORCE_SUBMIT if last else "auto",
            max_tokens=limits.max_tokens_per_response,
            extra=extra,
        )
        # Counted for cached responses too, so a replay stops where the live run did.
        usage = usage + response.usage
        step = AgentStep(index=i, response=response, forced_submit=last)
        steps.append(step)
        messages.append(response.assistant_message())

        if not response.tool_calls:
            step.nudged = True
            messages.append({"role": "user", "content": NUDGE})
        for call in response.tool_calls:
            outcome = execute_tool_call(registry, call)
            step.tool_outcomes.append(outcome)
            if call.name == SUBMIT and outcome.error is None:
                args = SubmitAnswerArgs.model_validate(outcome.arguments)
                problems = validate_submission(args, ledger, check_figures=check_figures)
                if not problems:
                    final = finalize(args, ledger, limits)
                    stop = "submitted"
                    break
                rejections += 1
                outcome.output = None
                outcome.error = "answer rejected: " + " | ".join(problems)
                if rejections >= limits.max_rejections:
                    final = FinalAnswer(
                        status="needs_review",
                        answer=args.answer,
                        confidence=args.confidence,
                        rationale=args.rationale,
                        citations=[c.model_dump() for c in args.citations],
                        review_reasons=[f"answer failed validation {rejections} times", *problems],
                    )
                    stop = "validation_failed"
                    break
            else:
                for ev in outcome.evidence:
                    ledger.add(ev)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.content})
        if final is not None:
            break
        if usage.total_tokens >= limits.max_tokens_per_run:
            stop = "token_budget"
            break

    if final is None:
        final = FinalAnswer(
            status="needs_review",
            answer=None,
            confidence=None,
            rationale=None,
            citations=[],
            review_reasons=[f"run ended without a valid answer ({stop})"],
        )
    return AgentResult(
        question=question,
        final=final,
        stop_reason=stop,
        steps=steps,
        usage=usage,
        messages=messages,
        rejections=rejections,
        evidence=ledger,
    )
