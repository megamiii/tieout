# tieout

An AI agent that answers audit-style questions about public companies' SEC filings, cites every source, and checks that the numbers in a filing actually tie out.

**[Live demo](TODO-demo-url)** · works without an API key (demo mode)

In auditing, a *tie-out* is the check that every number stated in a document matches the underlying financial records. tieout automates a first pass of that work on real 10-K filings. An agent searches the filing text, pulls structured financial data from the SEC's XBRL API, does arithmetic with a real calculator instead of the LLM, and returns an answer with citations and a confidence score. When the evidence is missing or contradictory, it flags the question for human review instead of guessing.

<!-- TODO: add a GIF of the demo: question → agent steps → cited answer -->

## What you can try in the demo

- **Ask a question:** "What was 3M's FY2022 capital expenditure, and how did it compare to FY2021?"
- **Run a tie-out:** pick a filing, and the agent extracts numbers from the MD&A section and checks each one against the XBRL financial data, flagging mismatches
- **Analytical review:** the agent finds line items with large year-over-year changes and searches the filing for management's explanation
- **Inspect the trace:** see every tool call, retrieved passage, token count, cost and latency
- **Review queue:** accept or override low-confidence answers, as a human reviewer would

## Features

- **Agentic workflow:** an LLM agent with native tool calling decides which tools to use for each question
- **Hybrid retrieval (RAG):** vector + keyword search over chunked 10-K filings in Postgres with pgvector, using an open-source embedding model
- **Tools**
  - `search_filings`: hybrid search with section and page metadata
  - `read_section`: reads a full section of a filing
  - `get_financial_facts`: structured XBRL data from the SEC EDGAR API (cached)
  - `calculate`: safe arithmetic, so numbers are computed, not generated
  - `submit_answer`: schema-validated answer with citations
- **Grounded, explainable answers:** every claim links to the exact passage or XBRL fact it came from; answers with invalid citations are rejected and retried
- **Human-in-the-loop:** low-confidence or conflicting answers go to a review queue, and reviewer decisions are saved
- **Evaluation harness:** tested on the FinanceBench open-source benchmark plus numeric questions with ground truth from XBRL, run in CI
- **Observability:** a full trace of every agent run
- **MCP server:** the filing search and financial data tools are also exposed through the Model Context Protocol for AI assistants and IDEs that support it
- **Demo mode:** curated questions replay recorded agent runs, so anyone can try the demo for free; live mode uses a rate-limited, budget-capped API key

## Architecture

<!-- TODO: replace with the final diagram -->

```
            ┌──────────────────────────── Web UI (React + TypeScript) ────────────────────────────┐
            │  ask · tie-out · analytical review · agent trace · review queue · eval dashboard    │
            └──────────────────────────────────────┬──────────────────────────────────────────────┘
                                                   │ REST API
                        ┌──────────────────────────▼──────────────────────────┐
                        │ FastAPI backend                                      │
                        │   Agent loop ──► LLM API (Groq / Ollama)             │
                        │     ├─ search_filings ──► hybrid RAG                 │
                        │     ├─ read_section                                  │
                        │     ├─ get_financial_facts ──► SEC EDGAR XBRL (cache)│
                        │     ├─ calculate                                     │
                        │     └─ submit_answer (schema + citation checks)      │
                        │   Demo mode (recorded runs) · tracing · cost limits  │
                        └──────────────────────────┬──────────────────────────┘
                                                   │
                        ┌──────────────────────────▼──────────────────────────┐
                        │ Postgres + pgvector                                  │
                        │ filings · chunks · XBRL facts · answers · traces     │
                        └──────────────────────────▲──────────────────────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        │ MCP server (same tools, for external assistants)     │
                        └─────────────────────────────────────────────────────┘
```

**Design decisions**

<!-- TODO: fill in as you build. Explain each choice in your own words. -->

- **Why a calculator tool:** _LLMs make arithmetic mistakes; financial answers must be computed_
- **Text plus XBRL:** _why combining unstructured filing text with structured financial data gives better answers than either alone_
- **Hybrid retrieval:** _why keyword search matters for line-item names, fiscal years and exact figures_
- **When to abstain:** _how confidence is estimated, and why "needs review" beats a wrong answer in audit work_
- **Demo mode:** _how recorded runs keep the public demo free and reliable_

Data layer (milestone 1):

- **Download once, politely:** every download goes through one HTTP client with a permanent disk cache (URL → readable file path, written atomically), a rate limiter shared by all SEC hosts (5 req/s by default, capped at the SEC's 10 req/s), a declared User-Agent from `.env`, and retries with exponential backoff + jitter on 429/5xx that honor `Retry-After`. A second `tieout data download` makes 0 network requests.
- **Company selection is a rule, not a hand-picked list:** every company with at least 4 FinanceBench questions about a 10-K (13 companies, 70 questions). The output, `config/companies.json`, is committed so it can be reviewed.
- **Filing text from FinanceBench PDFs:** FinanceBench evidence cites PDF page numbers, so parsing the same PDFs lets the eval check cited pages against the gold pages. EDGAR supplies each PDF's accession number.
- **Fiscal year → filing by period end date, not the SEC's `fy` field:** the companyfacts `fy` is unreliable (Best Buy's 10-K for the year ending 2021-01-30 is tagged `fy=2020`). Filings are matched by period end, with original 10-Ks preferred over 10-K/As, and cross-checked against the accession number when the FinanceBench link contains one.
- **Facts are tied to the filing that reported them:** each 10-K's XBRL repeats prior years as comparatives, so a fact is stored per (concept, period, accession). A tie-out of the FY2018 10-K compares against the values *that* filing reported, not later restatements.
- **Stable IDs:** XBRL fact IDs are a hash of (CIK, concept, unit, period, accession), so recorded demo runs and eval reports that cite `xf_…` IDs survive a database rebuild.
- **Bulk load with `COPY`:** loading the 135,144 facts with batched ORM `INSERT`s was still unfinished after 10 minutes. Postgres `COPY` loads them in about 12 s, inside the same transaction that deletes the company's old rows, so a reload is idempotent.

Parsing and retrieval (milestone 2):

- **Page numbers are verified, not assumed:** FinanceBench's `evidence_page_num` is a 0-based PDF page index. All 90 gold evidence passages for our filings best-match exactly that page in our PyMuPDF extraction, so gold page = index + 1 in our 1-based numbering. The cover-page "fiscal year ended" date matches EDGAR's period end for all 32 filings.
- **Section detection that survives real PDFs:** item headings are found with a line-anchored regex. Table-of-contents pages (5+ headings in the first 15% of the document) are skipped. A heading must contain its item's title keyword, compared letters-only because extraction splits words ("Busines s"). Comma cross-references ("Item 7, Management's Discussion…, and the Notes") are rejected, and items must appear in increasing order. Item 7 (MD&A) spans 3+ pages in 31/32 filings and Item 8 in 26/32. The misses are real document structure: PepsiCo includes its financial statements in Item 7, Corning puts them after Item 15, and CVS 2018 incorporates both from an appended exhibit.
- **Chunks never cross a page or section**, so every citation is an exact (filing, item, page). Chunks are built from whole lines, which keeps table cells intact.
- **Chunk size is counted in model tokens, not words:** bge-small silently truncates input past 512 tokens, and financial tables tokenize densely ("1,577" is 3 tokens). With 300-word chunks, 4.7% of chunks overflowed (one was 4,360 tokens). Sizing with the model's own tokenizer (max 400 tokens + context header, 50 tokens overlap) brought overflow to 0%.
- **Context header:** each chunk is embedded and indexed as "3M (MMM) 10-K FY2018 | Item 8: Financial Statements… | page 60" + text, so a table fragment of bare numbers still says what it belongs to.
- **Free local embeddings:** BAAI/bge-small-en-v1.5 runs through fastembed (ONNX Runtime, no PyTorch, which keeps the deployment image small). On a 15 W laptop CPU (i7-10510U), embedding the 13,003 chunks took about 70 minutes. Embeddings are cached per filing, keyed by a hash of model + text, so a full rebuild takes under 2 minutes and embeds nothing. In production only the user's short query is embedded.
- **Filters inside both retrievers:** company, fiscal year, section and filing filters are SQL `WHERE` clauses in both the vector and the keyword query, not post-filters. pgvector's HNSW index applies `WHERE` after the approximate search, so a selective filter (one filing is ~3% of rows) can return fewer than k rows. pgvector 0.8 iterative index scans keep scanning until k rows pass the filter.
- **Keyword search needs IDF:** Postgres `ts_rank_cd` has no inverse document frequency, so "statement" (in 44% of chunks) counted as much as "expenditure" (2.9%). Keyword search now uses a BM25-lite score: per-lexeme document frequencies from `ts_stat`, terms in more than 20% of chunks dropped as corpus stopwords, and score = sum of IDF over the query terms a chunk contains. On the retrieval eval this doubled keyword hit@1 (4.3% → 10.0%).
- **Measured result: on raw FinanceBench questions, vector search alone beats hybrid** (hit@10 68.6% vs 54.3%; see [Retrieval](#retrieval-milestone-2)). Keyword search mostly fails from vocabulary mismatch: questions say "capital expenditure" or "quick ratio", while the gold page says "Purchases of property, plant and equipment" or shows only the raw line items. Stripping FinanceBench's instruction sentences barely changes this (keyword hit@10 24.3% → 28.6%). RRF weights were not tuned on these 70 questions, to avoid fitting the method to the test set. Which mode the agent uses will be decided by the agent experiments in milestone 5, where the agent writes its own search queries.
- **Known limitation:** dense financial-statement tables retrieve poorly in every mode. For "3M FY2018 capital expenditure", the gold cash-flow-statement chunk (p. 60) is not in the top 10 of any retriever. Exact line-item values are what the XBRL tool (`get_financial_facts`) is for.

LLM client and agent loop (milestone 3):

- **Free tier shapes the design:** Groq's free tier for `openai/gpt-oss-20b` / `gpt-oss-120b` allows 30 requests/min, 1,000 requests/day, 8,000 tokens/min and 200,000 tokens/day (from Groq's rate-limit docs, checked 2026-10-07). An agent resends the whole conversation on every step, so tokens, not requests, are the binding limit. Search results are truncated to 1,000 characters per passage, the default is 5 passages, Pydantic's schema `title` fields are stripped from tool schemas, and each run has a token budget.
- **One client, any OpenAI-compatible provider:** a small `httpx` client for `/chat/completions`, so `LLM_PROVIDER=groq|ollama|openai_compatible` switches providers. It doesn't use an SDK, so retries, caching and token accounting are all visible in about 200 lines.
- **Response cache = recordings:** each request is serialized as canonical JSON (sorted keys, plus the provider URL) and hashed with SHA-256; the response is stored under that hash. Re-running an identical request costs no tokens. In offline mode the client serves only cached responses and raises on a miss, so a recorded run replays exactly, which is the mechanism for keyless tests and, later, the public demo.
- **Rate limiting before, retrying after:** a sliding-window limiter waits *before* a request that would exceed requests/min or tokens/min (estimated at ~4 characters per token). If a 429 still happens, the client retries after Groq's `retry-after`, using the same backoff code as the SEC downloader. A daily ledger (`data/usage/YYYY-MM-DD.json`) records tokens per model, cache hits, tokens saved and Groq's latest `x-ratelimit-*` headers; `tieout usage` reports it against the daily limits.
- **The loop is written by hand** (`backend/tieout/agent/loop.py`, no framework): send messages and tool schemas, execute each tool call, append each result as a `tool` message tied to its `tool_call_id`, and stop when the model answers in plain text.
- **Guardrails:** a maximum number of steps, where the last step sets `tool_choice="none"` so the model answers with what it has instead of the run ending empty; a token budget per run, counted for cached responses too so a replay stops exactly where the live run did; and tool errors (invalid JSON, schema violations, unknown tickers) returned to the model as `{"error": ...}` so it can correct itself.

## Data

All data is public:

- **Filings:** 32 10-K annual reports (FY2015–FY2023) for 13 companies, from the [FinanceBench](https://github.com/patronus-ai/financebench) document set and [SEC EDGAR](https://www.sec.gov/search-filings)
- **Structured financials:** XBRL company facts from the [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) (no API key; requests use a declared User-Agent and stay under the SEC's 10 requests/second limit)
- **Evaluation questions:** the 150-question open-source sample of FinanceBench, plus numeric questions generated from XBRL data with exact ground truth

**Supported companies** (selected by `tieout data select`: every company with ≥ 4 FinanceBench 10-K questions)

| Ticker | Company | Sector | 10-K questions | Fiscal years |
|---|---|---|---|---|
| AMD | AMD | Information Technology | 8 | 2015, 2022 |
| BA | Boeing | Industrials | 8 | 2018, 2022 |
| AXP | American Express | Financials | 7 | 2022 |
| PEP | PepsiCo | Consumer Staples | 6 | 2021, 2022 |
| MMM | 3M | Industrials | 5 | 2018, 2022 |
| ADBE | Adobe | Information Technology | 5 | 2015, 2016, 2017, 2022 |
| AMCR | Amcor | Materials | 5 | 2020, 2023 |
| BBY | Best Buy | Consumer Discretionary | 5 | 2017, 2019, 2023 |
| VZ | Verizon | Communication Services | 5 | 2021, 2022 |
| GLW | Corning | Information Technology | 4 | 2020, 2021, 2022 |
| CVS | CVS Health | Health Care | 4 | 2018, 2022 |
| GIS | General Mills | Consumer Staples | 4 | 2019, 2020, 2022 |
| NKE | Nike | Consumer Discretionary | 4 | 2018, 2019, 2021, 2023 |

Data volume after `tieout data all`: 32 PDFs + 13 companyfacts files (119.9 MB cached on disk); 135,144 XBRL facts from 10-K/10-K/A filings (2,604 distinct concepts) loaded into Postgres.

After `tieout data index`: 705 sections and 13,003 chunks (bge-small-en-v1.5, 384 dimensions). The local Postgres database is 254 MB (chunks 131 MB including HNSW and GIN indexes, XBRL facts 102 MB, sections 12 MB), within Neon's 0.5 GB free tier.

## Evaluation

<!-- TODO: fill in with real results -->

| Metric | Description | Result |
|---|---|---|
| FinanceBench accuracy | Correct answers on the open-source sample (graded by numeric tolerance or an LLM judge) | _TODO_ |
| XBRL numeric accuracy | Exact-match on generated numeric questions | _TODO_ |
| Citation precision | Cited passages actually support the answer | _TODO_ |
| Abstention quality | "Needs review" used for genuinely hard cases, not easy ones | _TODO_ |
| Cost per question | Average LLM cost | _TODO_ |
| Latency | p50 / p95 per question | _TODO_ |

**Experiments**

| Configuration | FinanceBench accuracy | Cost/question |
|---|---|---|
| Single prompt, vector RAG only (baseline) | _TODO_ | _TODO_ |
| Agent + hybrid RAG | _TODO_ | _TODO_ |
| Agent + hybrid RAG + XBRL + calculator | _TODO_ | _TODO_ |

### Retrieval (milestone 2)

`tieout eval retrieval` searches with each of the 70 FinanceBench questions about supported 10-Ks and checks whether any top-k chunk lies on a gold evidence page of the right filing. Full report: [evals/reports/retrieval.md](evals/reports/retrieval.md) (per-question results in `retrieval.json`). Run on 2026-10-07; all rows filter to the question's filing unless noted.

| Configuration | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| vector | 20.0% | 42.9% | 54.3% | 68.6% | 0.343 | 46 | 87 |
| keyword, ts_rank_cd (no IDF) | 4.3% | 10.0% | 15.7% | 25.7% | 0.093 | 56 | 182 |
| keyword, BM25-lite IDF | 10.0% | 17.1% | 17.1% | 24.3% | 0.138 | 82 | 130 |
| hybrid RRF, ts_rank_cd keyword | 12.9% | 27.1% | 32.9% | 50.0% | 0.227 | 109 | 268 |
| hybrid RRF, BM25-lite keyword | 11.4% | 28.6% | 37.1% | 54.3% | 0.234 | 121 | 217 |
| hybrid RRF, BM25-lite, no filing filter | 5.7% | 22.9% | 28.6% | 42.9% | 0.159 | 472 | 730 |

## Getting started

### Requirements

- Python 3.11+, Node.js 20+, Docker
- Optional: a free [Groq](https://console.groq.com) API key, or [Ollama](https://ollama.com) for local runs (neither is needed for demo mode). The project never needs a paid API key.

### Run locally

```bash
git clone https://github.com/megamiii/tieout.git
cd tieout
cp .env.example .env            # set SEC_USER_AGENT to "<name> <your email>"
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
docker compose up -d --wait     # Postgres + pgvector
tieout data all                 # select, download (cached), load XBRL, parse + chunk + embed + index
tieout search "capital expenditures" --ticker MMM --year 2018
tieout ask "What was 3M's FY2018 capital expenditure?"   # needs GROQ_API_KEY or Ollama
tieout usage                    # today's tokens vs the free-tier daily limits
pytest -q
```

The first `tieout data all` embeds 13,003 chunks on the CPU (about 70 minutes on a 15 W laptop CPU); later runs reuse the embedding cache and finish in a few minutes. `make ingest`, `make test` and `make lint` wrap the same commands. The web UI (`make dev`) arrives in a later milestone; see [Build progress](#build-progress).

### Evaluate

```bash
tieout eval retrieval           # retrieval hit@k / MRR -> evals/reports/retrieval.{md,json}
make eval                       # (milestone 5) runs the agent on the eval set and writes a report
```

### Use the MCP server

```bash
make mcp
```

_TODO: add the MCP client config snippet._

## Build progress

- [x] 1. Data + caching: FinanceBench sample, company selection, EDGAR filings + XBRL facts, cached downloads, Postgres load
- [x] 2. Parsing (10-K sections with page numbers) + hybrid retrieval, with a retrieval eval
- [x] 3. Agent loop with one tool (offline tests only so far; the first live run with a Groq key is still to do)
- [ ] 4. All tools + answer validation
- [ ] 5. Evals
- [ ] 6. UI + review queue
- [ ] 7. Tie-out + analytical review
- [ ] 8. Tracing
- [ ] 9. MCP server
- [ ] 10. Demo mode + deployment
- [ ] 11. CI + README polish

## Project structure

<!-- TODO: update to match the final layout -->

Current layout:

```
tieout/
├── backend/tieout/
│   ├── net/              # cached HTTP client, rate limiter, retry/backoff
│   ├── edgar/            # SEC EDGAR client, filing matching, XBRL flattening
│   ├── financebench/     # FinanceBench dataset + company selection rule
│   ├── db/               # SQLAlchemy models, engine, schema init
│   ├── parsing/          # PDF pages, 10-K section splitter, cover-page date
│   ├── retrieval/        # token-aware chunking, embeddings, hybrid search
│   ├── llm/              # OpenAI-compatible client, response cache, rate limits, usage ledger
│   ├── agent/            # agent loop, tools, system prompt
│   ├── evals/            # retrieval eval
│   ├── ingest.py         # select → download → load pipeline
│   ├── indexing.py       # parse → chunk → embed (cached) → index pipeline
│   └── cli.py            # `tieout` command line
├── config/companies.json # selected companies, filings and accession numbers
├── evals/reports/        # eval reports (Markdown + JSON)
├── tests/                # unit tests + DB tests (skipped without Postgres)
└── docker-compose.yml    # Postgres 16 + pgvector
```

Planned final layout:

```
tieout/
├── backend/
│   ├── agent/            # agent loop, tools, prompts, guardrails
│   ├── retrieval/        # ingestion, chunking, embeddings, hybrid search
│   ├── edgar/            # SEC EDGAR client, XBRL facts, caching
│   ├── api/              # FastAPI routes, demo mode
│   ├── tracing/          # step traces, token and cost tracking
│   └── mcp_server/       # MCP tools
├── frontend/             # React + TypeScript UI
├── evals/                # eval sets, graders, reports
├── tests/                # unit and integration tests
├── Dockerfile            # single image for deployment
└── .github/workflows/    # CI
```

## Roadmap

- [ ] 10-Q support and quarter-over-quarter analytical review
- [ ] Multi-agent setup with a separate critic agent
- [ ] Table extraction from filings for line-level tie-outs

## Data and license

The code is released under the MIT License. See [LICENSE](LICENSE).

SEC filings and XBRL data are public records from the U.S. Securities and Exchange Commission. FinanceBench is by Patronus AI and is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); it is used here for non-commercial evaluation only:

> P. Islam, A. Kannappan, D. Kiela, R. Qian, N. Scherrer, B. Vidgen. "FinanceBench: A New Benchmark for Financial Question Answering." arXiv:2311.11944, 2023.

## Author

[megamiii](https://github.com/megamiii) on GitHub
