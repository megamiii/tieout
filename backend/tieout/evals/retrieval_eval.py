"""Retrieval eval: do we retrieve FinanceBench's gold evidence pages?

For each FinanceBench question about a supported 10-K, search with the
question text and check whether any of the top-k chunks is on a gold evidence
page. FinanceBench page numbers are 0-based PDF page indexes (verified: all
90 evidence passages for our filings match that page exactly), so gold
page = evidence_page_num + 1 in our 1-based numbering.
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from tieout.financebench.dataset import Question
from tieout.financebench.selection import CompanyConfig
from tieout.retrieval.embeddings import Embedder
from tieout.retrieval.search import KeywordScoring, Mode, SearchFilters, search

KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class Config:
    name: str
    mode: Mode
    filter_filing: bool
    keyword_scoring: KeywordScoring = "idf"


CONFIGS = (
    Config("vector", "vector", True),
    Config("keyword, ts_rank_cd (no IDF)", "keyword", True, "ts_rank"),
    Config("keyword, BM25-lite IDF", "keyword", True),
    Config("hybrid RRF, ts_rank_cd keyword", "hybrid", True, "ts_rank"),
    Config("hybrid RRF, BM25-lite keyword", "hybrid", True),
    Config("hybrid RRF, BM25-lite, no filing filter", "hybrid", False),
)


@dataclass
class ConfigResult:
    name: str
    n: int
    hit_at: dict[int, float]
    mrr: float
    latency_ms_p50: float
    latency_ms_p95: float
    per_question: list[dict[str, object]] = field(default_factory=list)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def run_retrieval_eval(
    session: Session,
    embedder: Embedder,
    questions: list[Question],
    companies: list[CompanyConfig],
) -> list[ConfigResult]:
    filing_by_doc = {f.doc_name: f for c in companies for f in c.filings}
    cases = [q for q in questions if q.doc_name in filing_by_doc]
    results: list[ConfigResult] = []
    max_k = max(KS)
    for cfg in CONFIGS:
        hits = {k: 0 for k in KS}
        rr_total = 0.0
        latencies: list[float] = []
        per_q: list[dict[str, object]] = []
        for q in cases:
            gold = {e.evidence_page_num + 1 for e in q.evidence if e.doc_name == q.doc_name}
            filing = filing_by_doc[q.doc_name]
            filters = SearchFilters(accession=filing.accession) if cfg.filter_filing else None
            t0 = time.perf_counter()
            found = search(
                session,
                embedder,
                q.question,
                filters,
                k=max_k,
                mode=cfg.mode,
                keyword_scoring=cfg.keyword_scoring,
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            # A hit must come from the right filing (matters for the no-filter run).
            ranks = [
                i
                for i, h in enumerate(found, start=1)
                if h.accession == filing.accession and h.page in gold
            ]
            first = ranks[0] if ranks else None
            for k in KS:
                hits[k] += int(first is not None and first <= k)
            rr_total += 1 / first if first else 0.0
            per_q.append(
                {
                    "financebench_id": q.financebench_id,
                    "doc_name": q.doc_name,
                    "gold_pages": sorted(gold),
                    "retrieved_pages": [h.page for h in found],
                    "first_hit_rank": first,
                }
            )
        n = len(cases)
        results.append(
            ConfigResult(
                name=cfg.name,
                n=n,
                hit_at={k: hits[k] / n for k in KS},
                mrr=rr_total / n,
                latency_ms_p50=statistics.median(latencies),
                latency_ms_p95=percentile(latencies, 0.95),
                per_question=per_q,
            )
        )
    return results


def write_reports(results: list[ConfigResult], out_dir: Path, meta: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "results": [asdict(r) for r in results]}
    (out_dir / "retrieval.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Retrieval eval",
        "",
        f"Questions: {results[0].n} FinanceBench questions on supported 10-Ks. "
        "A hit means a top-k chunk is on a gold evidence page of the right filing. "
        "All configurations filter to the question's filing unless noted.",
        "",
        "| Configuration | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR | p50 ms | p95 ms |",
        "|---|" + "---|" * (len(KS) + 3),
    ]
    for r in results:
        cells = " | ".join(f"{r.hit_at[k]:.1%}" for k in KS)
        lines.append(
            f"| {r.name} | {cells} | {r.mrr:.3f} | {r.latency_ms_p50:.0f} | "
            f"{r.latency_ms_p95:.0f} |"
        )
    lines += ["", "Run metadata: " + ", ".join(f"{k}={v}" for k, v in meta.items()), ""]
    (out_dir / "retrieval.md").write_text("\n".join(lines), encoding="utf-8")
