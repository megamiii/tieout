"""Index filings: PDF -> sections -> chunks -> embeddings -> Postgres.

Embeddings are cached on disk per filing, keyed by a hash of model + text,
so rebuilding the database does not re-run the model.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sqlalchemy import delete
from sqlalchemy.orm import Session

from tieout.db.models import ChunkRow, SectionRow
from tieout.db.session import get_engine, get_sessionmaker, init_db
from tieout.financebench.selection import CompanyConfig, FilingConfig
from tieout.net.cached_client import CachedClient
from tieout.parsing.pdf import extract_pages
from tieout.parsing.sections import Section, split_sections
from tieout.retrieval.chunking import Chunk, DocMeta, chunk_sections
from tieout.retrieval.embeddings import Embedder, Vector
from tieout.retrieval.search import build_lexeme_stats

CHUNK_COLUMNS = (
    "chunk_id",
    "accession",
    "cik",
    "fiscal_year",
    "section",
    "section_title",
    "page",
    "ordinal",
    "header",
    "text",
    "embedding",
)


@dataclass
class IndexStats:
    filings: int = 0
    sections: int = 0
    chunks: int = 0
    embedded: int = 0  # chunks that needed the model (cache misses)


def parse_filing(pdf_path: Path) -> list[Section]:
    return split_sections(extract_pages(pdf_path))


class EmbeddingCache:
    """One .npz per filing: {key -> vector}, key = sha1(model + text)."""

    def __init__(self, root: Path, model_name: str) -> None:
        self.dir = root / model_name.replace("/", "__")
        self.model_name = model_name

    def key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model_name}\n{text}".encode()).hexdigest()

    def load(self, doc_name: str) -> dict[str, Vector]:
        path = self.dir / f"{doc_name}.npz"
        if not path.exists():
            return {}
        with np.load(path) as data:
            return {str(k): v for k, v in zip(data["keys"], data["vectors"], strict=True)}

    def save(self, doc_name: str, vectors: dict[str, Vector]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        keys = np.array(sorted(vectors))
        stacked = np.stack([vectors[k] for k in keys]) if len(keys) else np.zeros((0, 0))
        np.savez(self.dir / f"{doc_name}.npz", keys=keys, vectors=stacked)


def embed_chunks(
    chunks: list[Chunk], embedder: Embedder, cache: EmbeddingCache, doc_name: str
) -> tuple[list[Vector], int]:
    cached = cache.load(doc_name)
    keys = [cache.key(c.indexed_text) for c in chunks]
    missing = [i for i, k in enumerate(keys) if k not in cached]
    if missing:
        fresh = embedder.embed_documents([chunks[i].indexed_text for i in missing])
        for i, vec in zip(missing, fresh, strict=True):
            cached[keys[i]] = vec
        cache.save(doc_name, {k: cached[k] for k in keys})
    return [cached[k] for k in keys], len(missing)


def _write_filing(
    session: Session,
    company: CompanyConfig,
    filing: FilingConfig,
    sections: list[Section],
    chunks: list[Chunk],
    vectors: list[Vector],
) -> None:
    session.execute(delete(ChunkRow).where(ChunkRow.accession == filing.accession))
    session.execute(delete(SectionRow).where(SectionRow.accession == filing.accession))
    session.add_all(
        SectionRow(
            accession=filing.accession,
            item=s.item,
            title=s.title,
            page_start=s.page_start,
            page_end=s.page_end,
            text=s.text,
        )
        for s in sections
    )
    session.flush()
    raw = session.connection().connection.driver_connection
    assert isinstance(raw, psycopg.Connection)
    register_vector(raw)
    with raw.cursor().copy(f"COPY chunks ({', '.join(CHUNK_COLUMNS)}) FROM STDIN") as copy:
        for chunk, vec in zip(chunks, vectors, strict=True):
            copy.write_row(
                (
                    chunk.chunk_id,
                    filing.accession,
                    company.cik,
                    filing.fiscal_year,
                    chunk.section,
                    chunk.section_title,
                    chunk.page,
                    chunk.ordinal,
                    chunk.header,
                    chunk.text,
                    vec,
                )
            )


def run_index(
    http: CachedClient,
    companies: list[CompanyConfig],
    embedder: Embedder,
    cache_root: Path,
    on_filing: Callable[[str, int, int], None] | None = None,
) -> IndexStats:
    engine = get_engine()
    init_db(engine)
    cache = EmbeddingCache(cache_root, embedder.model_name)
    stats = IndexStats()
    for company in companies:
        for filing in company.filings:
            sections = parse_filing(http.path(filing.pdf_url))
            meta = DocMeta(filing.doc_name, company.name, company.ticker, filing.fiscal_year)
            chunks = chunk_sections(meta, sections, count=embedder.count_tokens)
            vectors, n_new = embed_chunks(chunks, embedder, cache, filing.doc_name)
            # One transaction per filing: a crash leaves earlier filings indexed.
            with get_sessionmaker(engine)() as session, session.begin():
                _write_filing(session, company, filing, sections, chunks, vectors)
            stats.filings += 1
            stats.sections += len(sections)
            stats.chunks += len(chunks)
            stats.embedded += n_new
            if on_filing:
                on_filing(filing.doc_name, len(chunks), n_new)
    with get_sessionmaker(engine)() as session, session.begin():
        build_lexeme_stats(session)
    return stats
