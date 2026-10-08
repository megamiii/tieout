"""Database schema (SQLAlchemy 2.0 typed ORM)."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    cik: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(200))
    ticker: Mapped[str] = mapped_column(String(16), unique=True)
    sector: Mapped[str] = mapped_column(String(100))


class Filing(Base):
    __tablename__ = "filings"

    accession: Mapped[str] = mapped_column(String(20), primary_key=True)
    cik: Mapped[int] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"))
    doc_name: Mapped[str] = mapped_column(String(100), unique=True)  # FinanceBench name
    fiscal_year: Mapped[int] = mapped_column(Integer)
    form: Mapped[str] = mapped_column(String(10))
    period_end: Mapped[date] = mapped_column(Date)
    filing_date: Mapped[date] = mapped_column(Date)
    edgar_url: Mapped[str] = mapped_column(Text)
    pdf_url: Mapped[str] = mapped_column(Text)

    __table_args__ = (Index("ix_filings_cik_year", "cik", "fiscal_year"),)


class XbrlFactRow(Base):
    """One reported value from a 10-K's XBRL data (see tieout.edgar.xbrl)."""

    __tablename__ = "xbrl_facts"

    fact_id: Mapped[str] = mapped_column(String(19), primary_key=True)
    cik: Mapped[int] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"))
    taxonomy: Mapped[str] = mapped_column(String(16))
    concept: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(Text)
    unit: Mapped[str] = mapped_column(String(32))
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    value: Mapped[Decimal] = mapped_column(Numeric)
    # Not a foreign key: facts come from every 10-K, not only the filings we parse.
    accession: Mapped[str] = mapped_column(String(20))
    form: Mapped[str] = mapped_column(String(10))
    filed: Mapped[date] = mapped_column(Date)
    fiscal_year: Mapped[int | None] = mapped_column(Integer)
    fiscal_period: Mapped[str | None] = mapped_column(String(4))
    frame: Mapped[str | None] = mapped_column(String(16))

    __table_args__ = (
        Index("ix_xbrl_cik_concept_end", "cik", "concept", "period_end"),
        Index("ix_xbrl_accession", "accession"),
    )


class SectionRow(Base):
    """A 10-K item (e.g. Item 7 MD&A) with its full text, for read_section."""

    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    accession: Mapped[str] = mapped_column(ForeignKey("filings.accession", ondelete="CASCADE"))
    item: Mapped[str] = mapped_column(String(8))  # "1A", "7", ... or "front"
    title: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    __table_args__ = (Index("ix_sections_accession_item", "accession", "item", unique=True),)


class ChunkRow(Base):
    """A retrieval unit: part of one page of one section, with its embedding."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(19), primary_key=True)
    accession: Mapped[str] = mapped_column(ForeignKey("filings.accession", ondelete="CASCADE"))
    # Denormalized filter columns, so search needs no joins.
    cik: Mapped[int] = mapped_column(BigInteger)
    fiscal_year: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(8))
    section_title: Mapped[str] = mapped_column(Text)
    page: Mapped[int] = mapped_column(Integer)
    ordinal: Mapped[int] = mapped_column(Integer)
    header: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Full-text index over header + text, maintained by Postgres itself.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', header || ' ' || text)", persisted=True),
    )

    __table_args__ = (
        Index("ix_chunks_filter", "cik", "fiscal_year", "section"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# Row in lexeme_stats holding the total number of chunks (for IDF).
TOTAL_KEY = "__total__"


class LexemeStat(Base):
    """Document frequency of each full-text lexeme over all chunks (for IDF)."""

    __tablename__ = "lexeme_stats"

    word: Mapped[str] = mapped_column(Text, primary_key=True)
    ndoc: Mapped[int] = mapped_column(Integer)


class XbrlConcept(Base):
    """Catalog of XBRL concepts (label + description), embedded for lookup by meaning."""

    __tablename__ = "xbrl_concepts"

    concept: Mapped[str] = mapped_column(String(255), primary_key=True)
    taxonomy: Mapped[str] = mapped_column(String(16))
    label: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class AgentRun(Base):
    """A question answered by the agent (live, or imported from an eval run)."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(64))  # "live", "eval:agent_full", ...
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), index=True)  # answered | needs_review
    answer: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    review_reasons: Mapped[list[str]] = mapped_column(JSONB)
    warnings: Mapped[list[str]] = mapped_column(JSONB)
    stop_reason: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    requests: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    trace: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    eval_item_id: Mapped[str | None] = mapped_column(String(128), unique=True)


class ReviewDecision(Base):
    """A reviewer's decision on a run. Append-only: the latest decision is current."""

    __tablename__ = "review_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decision: Mapped[str] = mapped_column(String(16))  # accept | override
    final_answer: Mapped[str] = mapped_column(Text)
    note: Mapped[str] = mapped_column(Text)
    reviewer: Mapped[str] = mapped_column(String(100))
