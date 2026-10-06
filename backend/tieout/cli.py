"""Command-line entry point: `tieout --help`."""

import typer

from tieout import ingest
from tieout.config import get_settings

app = typer.Typer(no_args_is_help=True, help="tieout: audit agent for 10-K filings")
data_app = typer.Typer(no_args_is_help=True, help="Download and load source data.")
app.add_typer(data_app, name="data")


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


@data_app.command("all")
def data_all() -> None:
    """select + download + load."""
    data_select()
    data_download()
    data_load()


if __name__ == "__main__":
    app()
