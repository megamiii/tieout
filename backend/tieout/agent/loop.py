"""The agent loop, written by hand (no framework).

    messages = [system, user]
    repeat up to max_steps:
        response = LLM(messages, tool schemas)
        append the assistant message
        if it has no tool calls: that is the answer, stop
        run each tool call, append each result as a `tool` message
        stop if the run has used more than its token budget

On the last allowed step the request sets tool_choice="none", which makes the
model answer with what it has instead of the run ending with no answer.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from tieout.agent.tools import Tool, ToolOutcome, execute_tool_call
from tieout.llm.types import ChatResponse, Message, Usage

StopReason = Literal["answered", "max_steps", "token_budget"]


class ChatModel(Protocol):
    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> ChatResponse: ...


@dataclass
class AgentStep:
    index: int
    response: ChatResponse
    tool_outcomes: list[ToolOutcome] = field(default_factory=list)
    forced_answer: bool = False  # tool_choice="none" on this step


@dataclass
class AgentResult:
    question: str
    answer: str | None
    stop_reason: StopReason
    steps: list[AgentStep]
    usage: Usage
    messages: list[Message]

    @property
    def tool_calls(self) -> int:
        return sum(len(s.tool_outcomes) for s in self.steps)

    @property
    def cached(self) -> bool:
        """True if every LLM response came from the cache (no tokens spent)."""
        return all(s.response.cached for s in self.steps)


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 6
    max_tokens_per_run: int = 30_000
    max_tokens_per_response: int = 1024


DEFAULT_LIMITS = AgentLimits()


def run_agent(
    llm: ChatModel,
    tools: list[Tool],
    system_prompt: str,
    question: str,
    limits: AgentLimits = DEFAULT_LIMITS,
    extra: dict[str, Any] | None = None,
) -> AgentResult:
    registry = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    steps: list[AgentStep] = []
    usage = Usage()
    stop: StopReason = "max_steps"
    answer: str | None = None

    for i in range(limits.max_steps):
        last = i == limits.max_steps - 1
        response = llm.chat(
            messages,
            tools=schemas,
            tool_choice="none" if last else "auto",
            max_tokens=limits.max_tokens_per_response,
            extra=extra,
        )
        # Budget counts tokens even for cached responses, so a replayed run
        # stops exactly where the live run did.
        usage = usage + response.usage
        step = AgentStep(index=i, response=response, forced_answer=last)
        steps.append(step)
        messages.append(response.assistant_message())

        if not response.tool_calls:
            answer = response.content
            stop = "answered"
            break
        for call in response.tool_calls:
            outcome = execute_tool_call(registry, call)
            step.tool_outcomes.append(outcome)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.content})
        if usage.total_tokens >= limits.max_tokens_per_run:
            stop = "token_budget"
            break

    return AgentResult(
        question=question,
        answer=answer,
        stop_reason=stop,
        steps=steps,
        usage=usage,
        messages=messages,
    )
