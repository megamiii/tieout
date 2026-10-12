from functools import lru_cache

import pytest


@lru_cache(maxsize=1)
def postgres_error() -> str | None:
    """Why Postgres is unreachable, checked once per test run (None if it's up).

    A short connect timeout matters: on some networks a closed port drops the
    connection instead of refusing it, and every DB test would hang for minutes.
    """
    from sqlalchemy import create_engine, text

    from tieout.config import get_settings
    from tieout.db.session import sqlalchemy_url

    engine = create_engine(
        sqlalchemy_url(get_settings().database_url), connect_args={"connect_timeout": 3}
    )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as err:
        return type(err).__name__
    finally:
        engine.dispose()
    return None


@pytest.fixture(autouse=True)
def _skip_db_tests_without_postgres(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("db") and (err := postgres_error()):
        pytest.skip(f"Postgres not running (docker compose up -d): {err}")
