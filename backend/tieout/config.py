"""Application settings, read from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: backend/tieout/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    sec_user_agent: str = ""
    # The SEC allows at most 10 requests/second; never configure above that.
    sec_max_rps: float = Field(default=5.0, gt=0, le=10)

    database_url: str = "postgresql+psycopg://tieout:tieout@localhost:5432/tieout"
    data_dir: Path = Path("data")

    # LLM: any OpenAI-compatible chat completions API. Free options only:
    # Groq or Gemini free tiers, or a local Ollama server.
    llm_provider: str = "groq"  # groq | gemini | ollama | openai_compatible
    llm_model: str = "openai/gpt-oss-20b"
    llm_base_url: str = ""  # overrides the provider preset
    groq_api_key: str = ""
    gemini_api_key: str = ""
    llm_api_key: str = ""  # for openai_compatible providers
    # Free-tier limits, when the provider doesn't publish fixed ones (Gemini shows
    # each project's limits in AI Studio). 0 = use the built-in preset, if any.
    llm_rpm: int = Field(default=0, ge=0)  # requests per minute
    llm_tpm: int = Field(default=0, ge=0)  # tokens per minute
    llm_rpd: int = Field(default=0, ge=0)  # requests per day
    llm_tpd: int = Field(default=0, ge=0)  # tokens per day
    # gpt-oss reasoning effort (low | medium | high); empty = provider default.
    # Reasoning tokens count against the free-tier token limits.
    llm_reasoning_effort: str = ""

    # Agent guardrails
    agent_max_steps: int = Field(default=6, ge=1, le=20)
    agent_max_tokens_per_run: int = Field(default=30_000, gt=0)

    @property
    def data_path(self) -> Path:
        path = self.data_dir if self.data_dir.is_absolute() else REPO_ROOT / self.data_dir
        return path

    @property
    def cache_dir(self) -> Path:
        return self.data_path / "cache"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_path / "llm_cache"

    @property
    def usage_dir(self) -> Path:
        return self.data_path / "usage"


@lru_cache
def get_settings() -> Settings:
    return Settings()
