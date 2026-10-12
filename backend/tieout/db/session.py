"""Engine and session helpers."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from tieout.config import get_settings
from tieout.db.models import Base


def sqlalchemy_url(url: str) -> str:
    """Use the psycopg 3 driver for plain postgres URLs (as Neon hands them out)."""
    for scheme in ("postgresql://", "postgres://"):
        if url.startswith(scheme):
            return "postgresql+psycopg://" + url[len(scheme) :]
    return url


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    return create_engine(sqlalchemy_url(url or get_settings().database_url), pool_pre_ping=True)


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


def init_db(engine: Engine | None = None) -> None:
    """Create the pgvector extension and all tables (idempotent)."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
