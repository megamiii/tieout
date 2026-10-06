"""Provider-neutral types for OpenAI-compatible chat completions."""

from dataclasses import dataclass, field
from typing import Any

# Messages use the OpenAI chat format as plain dicts:
#   {"role": "system" | "user" | "assistant" | "tool", "content": ..., ...}
Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: Usage
    model: str
    request_hash: str
    cached: bool  # served from the response cache: no tokens spent
    latency_ms: float
    reasoning: str | None = None  # reasoning text, for models that return it
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def assistant_message(self) -> Message:
        """The message to append to the conversation for this response."""
        msg: Message = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return msg


class LLMError(Exception):
    """A non-retryable error from the LLM API (bad request, auth, ...)."""


class LLMCacheMiss(Exception):
    """Offline mode: the request is not in the recorded cache."""
