"""Hybrid search against a real Postgres + pgvector (skipped if unavailable).

Rows are inserted inside a transaction that is always rolled back, so the
test never touches the real index. A tiny bag-of-words "embedder" makes
vector results predictable without loading a model.
"""

from collections.abc import Iterator, Sequence
from datetime import date

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from tieout.db.models import EMBEDDING_DIM, ChunkRow, Company, Filing
from tieout.db.session import get_engine, init_db
from tieout.retrieval.embeddings import Vector
from tieout.retrieval.search import SearchFilters, search

pytestmark = pytest.mark.db

VOCAB = ["revenue", "capital", "expenditure", "risk", "cyber", "dividend", "lease", "tax"]


class BagOfWordsEmbedder:
    model_name = "test-bow"
    dim = EMBEDDING_DIM

    def _vec(self, text_: str) -> Vector:
        v = np.zeros(self.dim, dtype=np.float32)
        for i, word in enumerate(VOCAB):
            if word in text_.lower():
                v[i] = 1.0
        v[-1] = 0.01  # avoid zero vectors
        return v / np.linalg.norm(v)

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text_: str) -> Vector:
        return self._vec(text_)

    def count_tokens(self, text_: str) -> int:
        return len(text_.split())


@pytest.fixture
def session() -> Iterator[Session]:
    engine = get_engine()
    try:
        init_db(engine)
    except OperationalError:
        pytest.skip("Postgres not running (docker compose up -d)")
    conn = engine.connect()
    trans = conn.begin()
    s = Session(bind=conn)
    try:
        yield s
    finally:
        s.close()
        trans.rollback()
        conn.close()


CIK = 9_999_999_001
EMB = BagOfWordsEmbedder()


def add_chunk(
    s: Session, chunk_id: str, accession: str, year: int, section: str, body: str
) -> None:
    s.add(
        ChunkRow(
            chunk_id=chunk_id,
            accession=accession,
            cik=CIK,
            fiscal_year=year,
            section=section,
            section_title="t",
            page=1,
            ordinal=0,
            header=f"Testco 10-K FY{year}",
            text=body,
            embedding=EMB.embed_documents([body])[0].tolist(),
        )
    )


@pytest.fixture
def corpus(session: Session) -> Session:
    session.add(Company(cik=CIK, name="Testco", ticker="TSTX9", sector="Test"))
    for acc, year in (("T-2022", 2022), ("T-2023", 2023)):
        session.add(
            Filing(
                accession=acc,
                cik=CIK,
                doc_name=f"TESTCO_{year}_10K",
                fiscal_year=year,
                form="10-K",
                period_end=date(year, 12, 31),
                filing_date=date(year + 1, 2, 1),
                edgar_url="u",
                pdf_url="u",
            )
        )
    session.flush()
    add_chunk(session, "ch_t1", "T-2022", 2022, "7", "Capital expenditure was 1,577 million.")
    add_chunk(session, "ch_t2", "T-2022", 2022, "1A", "Cyber risk could harm operations.")
    add_chunk(session, "ch_t3", "T-2023", 2023, "7", "Capital expenditure rose to 1,749.")
    add_chunk(session, "ch_t4", "T-2023", 2023, "8", "Dividend and lease obligations.")
    session.flush()
    # Only our test company: other rows in the real index must not interfere.
    session.execute(text("SET LOCAL hnsw.ef_search = 100"))
    return session


def ids(hits: list) -> list[str]:  # type: ignore[type-arg]
    return [h.chunk_id for h in hits]


def test_filters_restrict_results(corpus: Session) -> None:
    f = SearchFilters(cik=CIK, fiscal_year=2023)
    hits = search(corpus, EMB, "capital expenditure", f, k=5)
    assert ids(hits)[0] == "ch_t3"
    assert all(h.fiscal_year == 2023 for h in hits)


def test_section_filter(corpus: Session) -> None:
    hits = search(corpus, EMB, "risk", SearchFilters(cik=CIK, section="1A"), k=5)
    assert ids(hits) == ["ch_t2"]


def test_keyword_mode_uses_or_semantics(corpus: Session) -> None:
    # No chunk contains every term, but OR semantics still finds partial matches.
    hits = search(
        corpus,
        EMB,
        "capital expenditure dividend zebra",
        SearchFilters(cik=CIK),
        k=5,
        mode="keyword",
    )
    assert set(ids(hits)) == {"ch_t1", "ch_t3", "ch_t4"}
    assert all(h.vector_rank is None for h in hits)


def test_hybrid_reports_both_ranks(corpus: Session) -> None:
    hits = search(corpus, EMB, "capital expenditure", SearchFilters(cik=CIK, fiscal_year=2022), k=2)
    top = hits[0]
    assert top.chunk_id == "ch_t1"
    assert top.vector_rank == 1 and top.keyword_rank == 1
    assert top.score == pytest.approx(2 / 61)
