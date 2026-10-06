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

    @property
    def data_path(self) -> Path:
        path = self.data_dir if self.data_dir.is_absolute() else REPO_ROOT / self.data_dir
        return path

    @property
    def cache_dir(self) -> Path:
        return self.data_path / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
