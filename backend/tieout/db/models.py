"""Database schema (SQLAlchemy 2.0 typed ORM)."""

from datetime import date
from decimal import Decimal

from sqlalchemy import BigInteger, Date, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
