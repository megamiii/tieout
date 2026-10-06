"""Wire an agent from settings: LLM client, tools and system prompt."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tieout.agent.loop import AgentLimits, AgentResult, ChatModel, run_agent
from tieout.agent.prompts import build_system_prompt
from tieout.agent.tools import SearchFilingsTool, Tool
from tieout.config import Settings
from tieout.db.session import get_sessionmaker
from tieout.financebench.selection import CompanyConfig
from tieout.llm.client import make_llm_client
from tieout.retrieval.embeddings import Embedder, get_embedder
from tieout.retrieval.search import Mode


@dataclass
class Agent:
    llm: ChatModel
    tools: list[Tool]
    system_prompt: str
    limits: AgentLimits
    extra: dict[str, Any] | None = None

    def ask(self, question: str) -> AgentResult:
        return run_agent(
            self.llm, self.tools, self.system_prompt, question, self.limits, self.extra
        )


def build_agent(
    settings: Settings,
    companies: list[CompanyConfig],
    *,
    llm: ChatModel | None = None,
    embedder: Embedder | None = None,
    search_mode: Mode = "hybrid",
    offline: bool = False,
    llm_cache_dir: Path | None = None,
) -> Agent:
    llm = llm or make_llm_client(settings, offline=offline, cache_dir=llm_cache_dir)
    embedder = embedder or get_embedder(settings.data_path / "models")
    tools: list[Tool] = [
        SearchFilingsTool(get_sessionmaker(), embedder, companies, mode=search_mode),
    ]
    extra = (
        {"reasoning_effort": settings.llm_reasoning_effort}
        if settings.llm_reasoning_effort
        else None
    )
    return Agent(
        llm=llm,
        tools=tools,
        system_prompt=build_system_prompt(companies),
        limits=AgentLimits(
            max_steps=settings.agent_max_steps,
            max_tokens_per_run=settings.agent_max_tokens_per_run,
        ),
        extra=extra,
    )
