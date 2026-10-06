"""Command-line entry point: `tieout --help`."""

from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import func, select

from tieout import ingest
from tieout.agent.factory import build_agent
from tieout.config import REPO_ROOT, get_settings
from tieout.db.models import ChunkRow
from tieout.db.session import get_sessionmaker
from tieout.evals.retrieval_eval import run_retrieval_eval, write_reports
from tieout.financebench.dataset import load_questions
from tieout.indexing import run_concepts, run_index
from tieout.llm.limits import UsageLedger, limits_for
from tieout.retrieval.chunking import MAX_TOKENS, OVERLAP_TOKENS
from tieout.retrieval.embeddings import get_embedder
from tieout.retrieval.search import RRF_K, SearchFilters, search

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="tieout: audit agent for 10-K filings",
)
data_app = typer.Typer(no_args_is_help=True, help="Download and load source data.")
app.add_typer(data_app, name="data")
eval_app = typer.Typer(no_args_is_help=True, help="Run evaluations.")
app.add_typer(eval_app, name="eval")


@data_app.command("select")
def data_select() -> None:
    """Pick supported companies from FinanceBench and write config/companies.json."""
    with ingest.client_from(get_settings()) as http:
        companies = ingest.run_select(http)
        n_questions = sum(c.question_count for c in companies)
        n_filings = sum(len(c.filings) for c in companies)
        for c in companies:
            years = ", ".join(str(f.fiscal_year) for f in c.filings)
            line = f"  {c.ticker:5} {c.name:20} CIK {c.cik:<8} {c.question_count:2} q"
            typer.echo(f"{line}  FY {years}")
        typer.echo(f"{len(companies)} companies, {n_filings} filings, {n_questions} questions")
        typer.echo(f"network requests: {http.stats.network}, cache hits: {http.stats.cache_hits}")


@data_app.command("download")
def data_download() -> None:
    """Download filing PDFs and XBRL company facts (cached; each file once)."""
    with ingest.client_from(get_settings()) as http:
        report = ingest.run_download(http, ingest.load_selected())
        typer.echo(
            f"{len(report.pdfs)} PDFs, {len(report.companyfacts)} companyfacts files, "
            f"{report.total_bytes() / 1e6:.1f} MB on disk"
        )
        typer.echo(f"network requests: {http.stats.network}, cache hits: {http.stats.cache_hits}")


@data_app.command("load")
def data_load() -> None:
    """Load companies, filings and XBRL facts into Postgres."""
    with ingest.client_from(get_settings()) as http:
        counts = ingest.run_load(http, ingest.load_selected())
        typer.echo(", ".join(f"{k}: {v}" for k, v in counts.items()))


@data_app.command("index")
def data_index() -> None:
    """Parse filings into sections, chunk, embed and load into Postgres."""
    settings = get_settings()
    embedder = get_embedder(settings.data_path / "models")
    with ingest.client_from(settings) as http:
        stats = run_index(
            http,
            ingest.load_selected(),
            embedder,
            settings.data_path / "embeddings",
            on_filing=lambda doc, n, new: typer.echo(f"  {doc}: {n} chunks, {new} embedded"),
        )
    typer.echo(
        f"{stats.filings} filings, {stats.sections} sections, {stats.chunks} chunks "
        f"({stats.embedded} newly embedded, {stats.chunks - stats.embedded} from cache)"
    )


@data_app.command("concepts")
def data_concepts() -> None:
    """Build the XBRL concept catalog (embedded labels + descriptions)."""
    settings = get_settings()
    embedder = get_embedder(settings.data_path / "models")
    with ingest.client_from(settings) as http:
        n = run_concepts(http, ingest.load_selected(), embedder, settings.data_path / "embeddings")
    typer.echo(f"{n} XBRL concepts in the catalog")


@data_app.command("all")
def data_all() -> None:
    """select + download + load + index."""
    data_select()
    data_download()
    data_load()
    data_concepts()
    data_index()


@app.command("search")
def search_cmd(
    query: str,
    ticker: Annotated[str | None, typer.Option(help="Company ticker, e.g. MMM")] = None,
    year: Annotated[int | None, typer.Option(help="Fiscal year")] = None,
    section: Annotated[str | None, typer.Option(help="10-K item, e.g. 7 or 1A")] = None,
    mode: Annotated[str, typer.Option(help="hybrid | vector | keyword")] = "hybrid",
    k: int = 5,
) -> None:
    """Search the indexed filings."""
    settings = get_settings()
    cik = None
    if ticker:
        by_ticker = {c.ticker: c.cik for c in ingest.load_selected()}
        cik = by_ticker[ticker.upper()]
    filters = SearchFilters(cik=cik, fiscal_year=year, section=section)
    embedder = get_embedder(settings.data_path / "models")
    with get_sessionmaker()() as session:
        hits = search(session, embedder, query, filters, k=k, mode=mode)  # type: ignore[arg-type]
    for h in hits:
        snippet = " ".join(h.text.split())[:160]
        typer.echo(
            f"{h.score:.4f} vec#{h.vector_rank} kw#{h.keyword_rank} {h.chunk_id} "
            f"FY{h.fiscal_year} Item {h.section} p.{h.page}\n    {snippet}"
        )


@eval_app.command("retrieval")
def eval_retrieval() -> None:
    """Hit@k / MRR of FinanceBench gold evidence pages; writes evals/reports/retrieval.*"""
    settings = get_settings()
    embedder = get_embedder(settings.data_path / "models")
    companies = ingest.load_selected()
    with ingest.client_from(settings) as http:
        questions = load_questions(http)
    with get_sessionmaker()() as session:
        results = run_retrieval_eval(session, embedder, questions, companies)
        n_chunks = session.scalar(select(func.count()).select_from(ChunkRow))
    meta = {
        "date": date.today().isoformat(),
        "embedding_model": embedder.model_name,
        "chunk_max_tokens": MAX_TOKENS,
        "chunk_overlap_tokens": OVERLAP_TOKENS,
        "chunks_indexed": n_chunks,
        "rrf_k": RRF_K,
    }
    write_reports(results, REPO_ROOT / "evals" / "reports", meta)
    typer.echo((REPO_ROOT / "evals" / "reports" / "retrieval.md").read_text(encoding="utf-8"))


@app.command("ask")
def ask_cmd(
    question: str,
    mode: Annotated[str, typer.Option(help="search mode: hybrid | vector | keyword")] = "hybrid",
    offline: Annotated[bool, typer.Option(help="replay recorded responses only")] = False,
    recording: Annotated[
        Path | None, typer.Option(help="recording dir (llm/ + tools/) to fill or replay")
    ] = None,
) -> None:
    """Ask the agent a question about the supported 10-K filings."""
    settings = get_settings()
    agent = build_agent(
        settings,
        ingest.load_selected(),
        search_mode=mode,  # type: ignore[arg-type]
        offline=offline,
        recording_dir=recording,
    )
    result = agent.ask(question)
    for step in result.steps:
        r = step.response
        src = "cache" if r.cached else f"{r.latency_ms:.0f} ms"
        flags = (" forced submit" if step.forced_submit else "") + (
            " (plain text, nudged)" if step.nudged else ""
        )
        typer.echo(
            f"[step {step.index + 1}] {r.usage.prompt_tokens}+{r.usage.completion_tokens} "
            f"tokens ({src}){flags}"
        )
        for o in step.tool_outcomes:
            status = f"error: {o.error}" if o.error else f"{len(o.evidence)} sources"
            typer.echo(f"    {o.name}({o.raw_arguments[:200]}) -> {status[:300]}")
    f = result.final
    typer.echo("")
    typer.echo(
        f"status: {f.status}  stop: {result.stop_reason}  tokens: {result.usage.total_tokens}"
    )
    if f.confidence is not None:
        typer.echo(f"confidence: {f.confidence:.2f}")
    typer.echo("")
    typer.echo(f.answer or "(no answer)")
    for c in f.citations:
        typer.echo(f"  [{c['source_id']}] {c['quote']!r}")
    for reason in f.review_reasons:
        typer.echo(f"  review: {reason}")
    for warning in f.warnings:
        typer.echo(f"  warning: {warning}")


@app.command("usage")
def usage_cmd() -> None:
    """Today's LLM token usage against the free-tier daily limits."""
    settings = get_settings()
    data = UsageLedger(settings.usage_dir).read()
    if not data:
        typer.echo("no LLM usage recorded today")
        return
    for model, m in data.items():
        tokens = m["prompt_tokens"] + m["completion_tokens"]
        lim = limits_for(settings.llm_provider, model)
        typer.echo(f"{model}:")
        if lim:
            typer.echo(f"  tokens   {tokens:>9,} / {lim.tpd:,} per day ({tokens / lim.tpd:.1%})")
            typer.echo(f"  requests {m['requests']:>9,} / {lim.rpd:,} per day")
        else:
            typer.echo(f"  tokens   {tokens:>9,} (no known daily limit)")
            typer.echo(f"  requests {m['requests']:>9,}")
        typer.echo(
            f"  cache hits {m['cache_hits']:,} (saved {m['tokens_saved_by_cache']:,} tokens)"
        )


if __name__ == "__main__":
    app()
