"""Command-line entry point: `tieout --help`."""

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
from sqlalchemy import func, select

from tieout import ingest
from tieout.agent.factory import build_agent
from tieout.config import REPO_ROOT, get_settings
from tieout.db.models import ChunkRow
from tieout.db.session import get_sessionmaker, init_db
from tieout.evals.metrics import compute, write_report
from tieout.evals.retrieval_eval import run_retrieval_eval, write_reports
from tieout.evals.runner import ARCHIVED, EXPERIMENTS, RUNS_DIR, run_answers, run_grading
from tieout.evals.sets import (
    EvalItem,
    build_financebench_set,
    build_xbrl_set,
    read_set,
    write_set,
)
from tieout.evals.tieout_eval import run_mutation_eval
from tieout.financebench.dataset import load_documents, load_questions
from tieout.indexing import run_concepts, run_index
from tieout.llm.client import configured_limits, make_llm_client, quota_timezone
from tieout.llm.limits import FreeTierLimits, UsageLedger, limits_for, quota_day
from tieout.retrieval.chunking import MAX_TOKENS, OVERLAP_TOKENS
from tieout.retrieval.embeddings import get_embedder
from tieout.retrieval.search import RRF_K, SearchFilters, search
from tieout.runs import import_eval_results
from tieout.workflows.service import run_analytical_review, run_tieout, save_workflow

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


SETS_DIR = REPO_ROOT / "evals" / "sets"
SET_NAMES = ("financebench", "xbrl_numeric")


def _load_sets(which: str) -> dict[str, list[EvalItem]]:
    names = SET_NAMES if which == "all" else (which,)
    return {n: read_set(SETS_DIR / f"{n}.jsonl") for n in names}


@eval_app.command("build-sets")
def eval_build_sets() -> None:
    """Write evals/sets/financebench.jsonl and evals/sets/xbrl_numeric.jsonl."""
    settings = get_settings()
    companies = ingest.load_selected()
    with ingest.client_from(settings) as http:
        fb = build_financebench_set(load_questions(http), load_documents(http), companies)
    with get_sessionmaker()() as session:
        xb = build_xbrl_set(session, companies)
    write_set(fb, SETS_DIR / "financebench.jsonl")
    write_set(xb, SETS_DIR / "xbrl_numeric.jsonl")
    n_num = sum(i.answer_kind == "numeric" for i in fb)
    typer.echo(f"financebench: {len(fb)} items ({n_num} numeric, {len(fb) - n_num} text)")
    typer.echo(f"xbrl_numeric: {len(xb)} items")


@eval_app.command("run")
def eval_run(
    experiment: Annotated[str, typer.Argument(help="baseline | agent_hybrid | agent_full")],
    eval_set: Annotated[
        str, typer.Option("--set", help="financebench | xbrl_numeric | all")
    ] = "all",
    limit: Annotated[int | None, typer.Option(help="answer at most N new items")] = None,
    offline: Annotated[bool, typer.Option(help="replay recordings only (no key/DB)")] = False,
    retry_errors: bool = False,
) -> None:
    """Answer (resumable) and grade one experiment. Re-run to continue."""
    settings = get_settings()
    companies = ingest.load_selected()
    if experiment not in EXPERIMENTS:
        raise typer.BadParameter(f"choose one of {list(EXPERIMENTS)}")
    for name, items in _load_sets(eval_set).items():

        def show(item_id: str, data: dict[str, Any] | None, _: str) -> None:
            d = data or {}
            status = d.get("error") or d.get("status")
            typer.echo(f"  {item_id}: {status} ({d.get('requests', 0)} requests)")

        typer.echo(f"[{experiment} / {name}] answering")
        s = run_answers(
            experiment,
            items,
            settings,
            companies,
            offline=offline,
            limit=limit,
            retry_errors=retry_errors,
            on_item=show,
        )
        typer.echo(f"  answered {s.done}, already done {s.skipped}, errors {s.errors}")
        g = run_grading(experiment, items, settings, offline=offline)
        typer.echo(f"  graded {g.done}, already graded {g.skipped}")
        for stop in (s.stopped, g.stopped):
            if stop:
                typer.echo(f"  stopped: {stop}")
                return


TIEOUT_EVAL_FILINGS = [("MMM", 2018), ("NKE", 2023), ("AMD", 2022), ("BA", 2022), ("GIS", 2022)]


@eval_app.command("tieout")
def eval_tieout(per_filing: int = 8) -> None:
    """Mutation test: plant 5-10% errors in matched MD&A claims; are they caught?"""
    settings = get_settings()
    embedder = get_embedder(settings.data_path / "models")
    client = make_llm_client(settings)
    with get_sessionmaker()() as session:
        result = run_mutation_eval(session, embedder, client, TIEOUT_EVAL_FILINGS, per_filing)
    out = REPO_ROOT / "evals" / "reports" / "tieout_mutation.json"
    out.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    typer.echo(
        f"planted {result['planted']}, detected {result['detected']} "
        f"({result['detection_rate']:.1%}), flagged against a different fact "
        f"{result['flagged_wrong_fact']}, outcomes {result['outcomes']}"
    )


@eval_app.command("report")
def eval_report() -> None:
    """Compute metrics from evals/runs and write evals/reports/eval.{md,json}."""
    settings = get_settings()
    sets = _load_sets("all")
    archived = [a for a in ARCHIVED if (RUNS_DIR / a).exists()]
    metrics = [
        compute(exp, name, items)
        for name, items in sets.items()
        for exp in [*EXPERIMENTS, *archived]
    ]
    meta = {
        "date": date.today().isoformat(),
        "agent_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "provider": settings.llm_provider,
    }
    path = write_report(metrics, REPO_ROOT / "evals" / "reports", meta)
    typer.echo(path.read_text(encoding="utf-8"))


runs_app = typer.Typer(no_args_is_help=True, help="Agent runs and the review queue.")
app.add_typer(runs_app, name="runs")


@runs_app.command("import-evals")
def runs_import_evals(
    experiment: Annotated[str, typer.Argument(help="eval experiment")] = "agent_full",
) -> None:
    """Load an experiment's eval results into the review queue (no LLM calls)."""
    settings = get_settings()
    items = [i for name in SET_NAMES for i in read_set(SETS_DIR / f"{name}.jsonl")]
    init_db()
    with get_sessionmaker()() as session, session.begin():
        n = import_eval_results(session, experiment, RUNS_DIR, items, settings.llm_model)
    typer.echo(f"imported {n} runs from evals/runs/{experiment}")


wf_app = typer.Typer(no_args_is_help=True, help="Audit workflows on one filing.")
app.add_typer(wf_app, name="workflow")


def _print_workflow(report: dict[str, Any]) -> None:
    typer.echo(
        f"{report['kind']} {report['ticker']} FY{report['fiscal_year']}: {report['summary']}"
    )


@wf_app.command("tieout")
def wf_tieout(
    ticker: str,
    fiscal_year: int,
    llm: Annotated[bool, typer.Option(help="adjudicate possible mismatches with the LLM")] = True,
    save: Annotated[bool, typer.Option(help="store the run for the UI")] = True,
    out: Annotated[Path | None, typer.Option(help="also write the report as JSON")] = None,
) -> None:
    """Tie out MD&A dollar claims against the filing's XBRL facts."""
    settings = get_settings()
    embedder = get_embedder(settings.data_path / "models")
    client = make_llm_client(settings) if llm else None
    init_db()
    with get_sessionmaker()() as session, session.begin():
        report = run_tieout(session, embedder, client, ticker, fiscal_year)
        if save:
            report["id"] = save_workflow(session, report, client.model if client else "none")
    _print_workflow(report)
    if out:
        out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")


@wf_app.command("review")
def wf_review(
    ticker: str,
    fiscal_year: int,
    save: Annotated[bool, typer.Option(help="store the run for the UI")] = True,
    out: Annotated[Path | None, typer.Option(help="also write the report as JSON")] = None,
) -> None:
    """Analytical review: material YoY changes and management's explanation."""
    settings = get_settings()
    embedder = get_embedder(settings.data_path / "models")
    client = make_llm_client(settings)
    init_db()
    with get_sessionmaker()() as session, session.begin():
        report = run_analytical_review(session, embedder, client, ticker, fiscal_year)
        if save:
            report["id"] = save_workflow(session, report, client.model)
    _print_workflow(report)
    if out:
        out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the API (and the built frontend, if frontend/dist exists)."""
    import uvicorn

    init_db()
    uvicorn.run("tieout.api.app:app", host=host, port=port, reload=reload)


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
    tz = quota_timezone(settings)
    data = UsageLedger(settings.usage_dir, today=quota_day(tz)).read()
    if not data:
        typer.echo("no LLM usage recorded today")
        return

    def against(used: int, limit: int | None) -> str:
        if not limit:
            return f"{used:>9,} (daily limit unknown; set it in .env)"
        return f"{used:>9,} / {limit:,} per day ({used / limit:.1%})"

    for model, m in data.items():
        tokens = m["prompt_tokens"] + m["completion_tokens"]
        if model == settings.llm_model:
            lim = configured_limits(settings)
        elif model == settings.judge_model:
            lim = FreeTierLimits(rpm=settings.judge_rpm, rpd=settings.judge_rpd, tpm=None, tpd=None)
        else:
            lim = limits_for(settings.llm_provider, model)
        typer.echo(f"{model}:")
        typer.echo(f"  tokens   {against(tokens, lim.tpd if lim else None)}")
        typer.echo(f"  requests {against(m['requests'], lim.rpd if lim else None)}")
        typer.echo(
            f"  cache hits {m['cache_hits']:,} (saved {m['tokens_saved_by_cache']:,} tokens)"
        )
    typer.echo(f"(quota day in {tz or 'local time'})")


if __name__ == "__main__":
    app()
