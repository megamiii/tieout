"""Hybrid search: pgvector similarity + Postgres full-text, fused with RRF.

Both retrievers apply the same metadata filters inside SQL, return their
top `candidates` chunk IDs, and reciprocal rank fusion merges the two
rankings: score(d) = sum over rankings of 1 / (RRF_K + rank(d)).
RRF only uses ranks, so it needs no calibration between cosine similarity
and ts_rank scores, which live on unrelated scales.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import ColumnElement, Text, cast, func, select, text
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.orm import Session

from tieout.db.models import ChunkRow
from tieout.retrieval.embeddings import Embedder

RRF_K = 60
Mode = Literal["hybrid", "vector", "keyword"]


@dataclass(frozen=True)
class SearchFilters:
    cik: int | None = None
    fiscal_year: int | None = None
    section: str | None = None
    accession: str | None = None

    def clauses(self) -> list[ColumnElement[bool]]:
        out: list[ColumnElement[bool]] = []
        if self.cik is not None:
            out.append(ChunkRow.cik == self.cik)
        if self.fiscal_year is not None:
            out.append(ChunkRow.fiscal_year == self.fiscal_year)
        if self.section is not None:
            out.append(ChunkRow.section == self.section)
        if self.accession is not None:
            out.append(ChunkRow.accession == self.accession)
        return out


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    accession: str
    fiscal_year: int
    section: str
    section_title: str
    page: int
    text: str
    score: float
    vector_rank: int | None  # 1-based rank in the vector list, None if absent
    keyword_rank: int | None


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of several ranked ID lists, best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # Tie-break by ID so results are deterministic.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def or_tsquery(query: str) -> ColumnElement[str]:
    """plainto_tsquery ANDs all terms; a question rarely has every term in one
    chunk, so OR them and let ts_rank_cd reward chunks matching more terms."""
    anded = cast(func.plainto_tsquery("english", query), Text)
    return cast(func.replace(anded, " & ", " | "), TSQUERY)


def vector_ids(
    session: Session, query_vec: Sequence[float], filters: SearchFilters, limit: int
) -> list[str]:
    # HNSW applies WHERE filters after the approximate search; with a selective
    # filter (one filing ~3% of rows) the default scan can return too few rows.
    # pgvector >= 0.8 iterative scans keep scanning until `limit` rows pass.
    session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    session.execute(text("SET LOCAL hnsw.ef_search = 100"))
    stmt = (
        select(ChunkRow.chunk_id)
        .where(*filters.clauses())
        .order_by(ChunkRow.embedding.cosine_distance(query_vec))
        .limit(limit)
    )
    return list(session.scalars(stmt))


def keyword_ids(session: Session, query: str, filters: SearchFilters, limit: int) -> list[str]:
    tsq = or_tsquery(query)
    stmt = (
        select(ChunkRow.chunk_id)
        .where(ChunkRow.tsv.op("@@")(tsq), *filters.clauses())
        .order_by(func.ts_rank_cd(ChunkRow.tsv, tsq).desc(), ChunkRow.chunk_id)
        .limit(limit)
    )
    return list(session.scalars(stmt))


def search(
    session: Session,
    embedder: Embedder,
    query: str,
    filters: SearchFilters | None = None,
    *,
    k: int = 8,
    mode: Mode = "hybrid",
    candidates: int = 50,
) -> list[SearchHit]:
    filters = filters or SearchFilters()
    vec_ids: list[str] = []
    kw_ids: list[str] = []
    if mode in ("hybrid", "vector"):
        vec_ids = vector_ids(session, embedder.embed_query(query).tolist(), filters, candidates)
    if mode in ("hybrid", "keyword"):
        kw_ids = keyword_ids(session, query, filters, candidates)

    fused = rrf([r for r in (vec_ids, kw_ids) if r])[:k]
    if not fused:
        return []
    rows = {
        r.chunk_id: r
        for r in session.scalars(
            select(ChunkRow).where(ChunkRow.chunk_id.in_([cid for cid, _ in fused]))
        )
    }
    vec_rank = {cid: i for i, cid in enumerate(vec_ids, start=1)}
    kw_rank = {cid: i for i, cid in enumerate(kw_ids, start=1)}
    return [
        SearchHit(
            chunk_id=cid,
            accession=rows[cid].accession,
            fiscal_year=rows[cid].fiscal_year,
            section=rows[cid].section,
            section_title=rows[cid].section_title,
            page=rows[cid].page,
            text=rows[cid].text,
            score=score,
            vector_rank=vec_rank.get(cid),
            keyword_rank=kw_rank.get(cid),
        )
        for cid, score in fused
    ]
