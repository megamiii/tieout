"""Wire an agent from settings: LLM client, tools and system prompt.

A *recording* directory holds both caches of a run: `llm/` (LLM responses)
and `tools/` (outputs of tools that read the database). Live runs fill it;
offline runs replay it with no API key, database or embedding model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tieout.agent.finance_tools import (
    CalculateTool,
    GetFinancialFactsTool,
    ReadSectionTool,
    SubmitAnswerTool,
)
from tieout.agent.loop import AgentLimits, AgentResult, ChatModel, run_agent
from tieout.agent.prompts import build_system_prompt
from tieout.agent.recording import RecordedTool
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


class _LazyEmbedder:
    """Loads the embedding model only if a tool actually needs it (not on replay)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.dim = 384

    def _get(self) -> Embedder:
        return get_embedder(self._settings.data_path / "models")

    def embed_documents(self, texts: Any) -> Any:
        return self._get().embed_documents(texts)

    def embed_query(self, text: str) -> Any:
        return self._get().embed_query(text)

    def count_tokens(self, text: str) -> int:
        return self._get().count_tokens(text)


def build_tools(
    settings: Settings,
    companies: list[CompanyConfig],
    *,
    embedder: Embedder | None = None,
    search_mode: Mode = "hybrid",
    use_xbrl: bool = True,
    recording_dir: Path | None = None,
    offline: bool = False,
) -> list[Tool]:
    embedder = embedder or _LazyEmbedder(settings)
    sessions = get_sessionmaker()
    data_tools: list[Tool] = [
        SearchFilingsTool(sessions, embedder, companies, mode=search_mode),
        ReadSectionTool(sessions, companies),
    ]
    if use_xbrl:
        data_tools.append(GetFinancialFactsTool(sessions, embedder, companies))
    tool_cache = recording_dir / "tools" if recording_dir else settings.data_path / "tool_cache"
    recorded: list[Tool] = [RecordedTool(t, tool_cache, offline=offline) for t in data_tools]
    pure: list[Tool] = [CalculateTool()] if use_xbrl else []
    return [*recorded, *pure, SubmitAnswerTool()]


def build_agent(
    settings: Settings,
    companies: list[CompanyConfig],
    *,
    llm: ChatModel | None = None,
    embedder: Embedder | None = None,
    search_mode: Mode = "hybrid",
    use_xbrl: bool = True,
    offline: bool = False,
    recording_dir: Path | None = None,
) -> Agent:
    llm_cache = recording_dir / "llm" if recording_dir else None
    llm = llm or make_llm_client(settings, offline=offline, cache_dir=llm_cache)
    tools = build_tools(
        settings,
        companies,
        embedder=embedder,
        search_mode=search_mode,
        use_xbrl=use_xbrl,
        recording_dir=recording_dir,
        offline=offline,
    )
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
