"""Run tie-out and analytical review for one filing against the database."""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from tieout.db.models import ChunkRow, Company, Filing, WorkflowRun, XbrlConcept, XbrlFactRow
from tieout.evals.grading import ChatModel
from tieout.llm.types import ChatResponse, Message
from tieout.retrieval.embeddings import Embedder
from tieout.retrieval.search import SearchFilters, search
from tieout.tracing.tracer import span
from tieout.workflows.analytical import explain, explanation_query, material_changes
from tieout.workflows.claims import Claim, dedupe, extract_claims
from tieout.workflows.tieout import (
    ClaimResult,
    FactLite,
    Vec,
    classify,
    concept_ranking,
    resolve,
    summarize,
)

PASSAGES_PER_CHANGE = 5


class CountingLLM:
    """Counts requests and tokens a workflow spends (cache hits cost nothing)."""

    def __init__(self, llm: ChatModel) -> None:
        self.llm = llm
        self.model = llm.model
        self.requests = 0
        self.cached = 0
        self.tokens = 0

    def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        resp = self.llm.chat(messages, **kwargs)
        if resp.cached:
            self.cached += 1
        else:
            self.requests += 1
            self.tokens += resp.usage.total_tokens
        return resp


def load_filing(session: Session, ticker: str, fiscal_year: int) -> tuple[Company, Filing]:
    company = session.scalar(select(Company).where(Company.ticker == ticker.upper()))
    if company is None:
        raise KeyError(f"unknown ticker {ticker!r}")
    filing = session.scalar(
        select(Filing).where(Filing.cik == company.cik, Filing.fiscal_year == fiscal_year)
    )
    if filing is None:
        raise KeyError(f"no {company.ticker} 10-K for FY{fiscal_year}")
    return company, filing


def filing_facts(session: Session, accession: str) -> list[FactLite]:
    rows = session.scalars(
        select(XbrlFactRow).where(XbrlFactRow.accession == accession, XbrlFactRow.unit == "USD")
    )
    return [
        FactLite(r.fact_id, r.concept, r.label, r.value, r.period_start, r.period_end) for r in rows
    ]


def concept_vectors(session: Session, concepts: set[str]) -> dict[str, Vec]:
    rows = session.scalars(select(XbrlConcept).where(XbrlConcept.concept.in_(concepts)))
    return {r.concept: np.asarray(r.embedding, dtype=np.float32) for r in rows}


def mdna_claims(session: Session, filing: Filing) -> list[Claim]:
    chunks = session.scalars(
        select(ChunkRow)
        .where(ChunkRow.accession == filing.accession, ChunkRow.section == "7")
        .order_by(ChunkRow.ordinal)
    )
    return dedupe(
        [
            c
            for ch in chunks
            for c in extract_claims(ch.chunk_id, ch.page, ch.text, filing.fiscal_year)
        ]
    )


def _fact_dict(f: FactLite | None) -> dict[str, Any] | None:
    if f is None:
        return None
    return {
        "fact_id": f.fact_id,
        "concept": f.concept,
        "label": f.label,
        "value": str(f.value),
        "period": f.period,
        "fiscal_year": f.fiscal_year,
    }


def _claim_item(r: ClaimResult) -> dict[str, Any]:
    c = r.claim
    return {
        "status": r.status,
        "amount": c.amount_text,
        "value": str(c.value),
        "tolerance": str(c.tolerance),
        "year": c.year,
        "chunk_id": c.chunk_id,
        "page": c.page,
        "sentence": c.sentence,
        "quarterly": c.quarterly,
        "non_gaap": c.non_gaap,
        "note": r.note,
        "fact": _fact_dict(r.fact),
        "concept_rank": r.concept_rank,
        "candidates": [_fact_dict(f) for f in r.candidates],
        "adjudication": r.adjudication,
    }


def run_tieout(
    session: Session,
    embedder: Embedder,
    llm: ChatModel | None,
    ticker: str,
    fiscal_year: int,
    claims: list[Claim] | None = None,
) -> dict[str, Any]:
    """Tie out the MD&A of one filing. `claims` overrides extraction (mutation tests)."""
    start = time.perf_counter()
    company, filing = load_filing(session, ticker, fiscal_year)
    claims = claims if claims is not None else mdna_claims(session, filing)
    facts = filing_facts(session, filing.accession)
    concepts = sorted({f.concept for f in facts})
    cvecs = concept_vectors(session, set(concepts))
    counter = CountingLLM(llm) if llm is not None else None
    results = []
    vectors: list[Vec] = (
        list(embedder.embed_documents([c.context for c in claims])) if claims else []
    )
    for claim, vec in zip(claims, vectors, strict=True):
        result = classify(claim, facts, concept_ranking(vec, concepts, cvecs))
        if result.status == "candidate_mismatch":
            span_input = {
                "amount": claim.amount_text,
                "year": claim.year,
                "sentence": claim.sentence,
                "candidates": [f.concept for f in result.candidates],
            }
            with span(
                "adjudication", f"{claim.amount_text} p.{claim.page}", input=span_input
            ) as sp:
                result = resolve(result, counter)
                sp.output = {
                    "status": result.status,
                    "note": result.note,
                    "reasoning": result.adjudication,
                }
        results.append(result)
    summary: dict[str, Any] = {**summarize(results)}
    summary.update(
        {
            "llm_requests": counter.requests if counter else 0,
            "llm_cached": counter.cached if counter else 0,
            "llm_tokens": counter.tokens if counter else 0,
            "seconds": round(time.perf_counter() - start, 1),
        }
    )
    return {
        "kind": "tieout",
        "ticker": company.ticker,
        "company": company.name,
        "fiscal_year": fiscal_year,
        "doc_name": filing.doc_name,
        "edgar_url": filing.edgar_url,
        "summary": summary,
        "items": [_claim_item(r) for r in results],
    }


def run_analytical_review(
    session: Session, embedder: Embedder, llm: ChatModel, ticker: str, fiscal_year: int
) -> dict[str, Any]:
    start = time.perf_counter()
    company, filing = load_filing(session, ticker, fiscal_year)
    facts = filing_facts(session, filing.accession)
    changes, base = material_changes(facts, fiscal_year)
    counter = CountingLLM(llm)
    items: list[dict[str, Any]] = []
    for ch in changes:
        with span("explain", ch.concept, input={"label": ch.label, "pct": str(ch.pct)}) as sp:
            # Explanations live in MD&A (Item 7); fall back to the whole filing for
            # filings whose MD&A sits elsewhere (e.g. an appended annual report).
            query = explanation_query(ch)
            hits = search(
                session,
                embedder,
                query,
                SearchFilters(accession=filing.accession, section="7"),
                k=PASSAGES_PER_CHANGE,
            ) or search(
                session,
                embedder,
                query,
                SearchFilters(accession=filing.accession),
                k=PASSAGES_PER_CHANGE,
            )
            passages = [
                {
                    "chunk_id": h.chunk_id,
                    "text": h.text,
                    "section": f"Item {h.section}",
                    "page": h.page,
                }
                for h in hits
            ]
            exp = explain(counter, ch, passages, fiscal_year)
            sp.output = {"status": exp.status, "quote": exp.quote, "chunk_id": exp.chunk_id}
        page = next((p["page"] for p in passages if p["chunk_id"] == exp.chunk_id), None)
        items.append(
            {
                "concept": ch.concept,
                "label": ch.label,
                "current": _fact_dict(ch.current),
                "prior": _fact_dict(ch.prior),
                "delta": str(ch.delta),
                "pct": str(round(ch.pct, 1)),
                "explanation": {
                    "status": exp.status,
                    "summary": exp.summary,
                    "chunk_id": exp.chunk_id,
                    "quote": exp.quote,
                    "page": page,
                },
                "passages": [p["chunk_id"] for p in passages],
            }
        )
    counts = {
        s: sum(i["explanation"]["status"] == s for i in items)
        for s in ("explained", "not_explained", "unverified")
    }
    return {
        "kind": "analytical_review",
        "ticker": company.ticker,
        "company": company.name,
        "fiscal_year": fiscal_year,
        "doc_name": filing.doc_name,
        "edgar_url": filing.edgar_url,
        "summary": {
            "material_changes": len(items),
            **counts,
            "materiality_base": base,
            "thresholds": {"min_change_pct": 20, "materiality_pct_of_base": 1},
            "llm_requests": counter.requests,
            "llm_cached": counter.cached,
            "llm_tokens": counter.tokens,
            "seconds": round(time.perf_counter() - start, 1),
        },
        "items": items,
    }


def save_workflow(session: Session, report: dict[str, Any], model: str) -> str:
    run_id = str(uuid.uuid4())
    session.add(
        WorkflowRun(
            id=run_id,
            created_at=datetime.now(UTC),
            kind=report["kind"],
            ticker=report["ticker"],
            fiscal_year=report["fiscal_year"],
            doc_name=report["doc_name"],
            model=model,
            summary=report["summary"],
            items=report["items"],
        )
    )
    return run_id


def workflow_dict(run: WorkflowRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "created_at": run.created_at.isoformat(),
        "kind": run.kind,
        "ticker": run.ticker,
        "fiscal_year": run.fiscal_year,
        "doc_name": run.doc_name,
        "model": run.model,
        "summary": run.summary,
        "items": run.items,
    }
