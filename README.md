# tieout

[![CI](https://github.com/megamiii/tieout/actions/workflows/ci.yml/badge.svg)](https://github.com/megamiii/tieout/actions/workflows/ci.yml) [![Pages](https://github.com/megamiii/tieout/actions/workflows/pages.yml/badge.svg)](https://github.com/megamiii/tieout/actions/workflows/pages.yml)

An AI agent that answers audit-style questions about public companies' SEC filings, cites every source, and checks that the numbers in a filing actually tie out.

**[Live demo](https://megamiii.github.io/tieout/)** · replays recorded runs, so it needs no API key and costs nothing

In auditing, a *tie-out* is the check that every number stated in a document matches the underlying financial records. tieout automates a first pass of that work on real 10-K filings. An agent searches the filing text, pulls structured financial data from the SEC's XBRL API, does arithmetic with a real calculator instead of the LLM, and returns an answer with citations and a confidence score. When the evidence is missing or contradictory, it flags the question for human review instead of guessing.

[![A recorded answer in the demo: what drove 3M's FY2022 operating margin, with the cited MD&A table and the quoted figures highlighted](docs/answer.png)](https://megamiii.github.io/tieout/#/ask?q=3m-operating-margin-drivers)

## What you can try in the demo

The public demo replays recorded runs, so it needs no API key and costs nothing (see the milestone 10 design notes under [Architecture](#architecture)).

- **Ask a question:** six recorded FinanceBench questions, for example [3M's FY2018 capital expenditure](https://megamiii.github.io/tieout/#/ask?q=3m-capex-2018) (XBRL value cross-checked against the cash flow statement), [Boeing's net income change](https://megamiii.github.io/tieout/#/ask?q=boeing-net-income-growth) (computed with the calculator), [what drove 3M's operating margin](https://megamiii.github.io/tieout/#/ask?q=3m-operating-margin-drivers) (quotes from the MD&A), and [an AMD liquidity question](https://megamiii.github.io/tieout/#/ask?q=amd-quick-ratio-review) the agent routed to human review
- **Run a tie-out:** five 10-Ks (3M FY2018, Nike FY2023, AMD FY2022, Boeing FY2022, General Mills FY2022): every dollar figure in the MD&A is checked against the filing's XBRL data, with exceptions flagged ([3M FY2018](https://megamiii.github.io/tieout/#/tieout?ticker=MMM&year=2018))
- **Analytical review:** line items with large year-over-year changes, and the MD&A passage that explains each one ([Nike FY2023](https://megamiii.github.io/tieout/#/analytical?ticker=NKE&year=2023); also 3M FY2018 and AMD FY2022)
- **Inspect the trace:** every LLM call, tool call and retrieval ranking, with tokens, latency and cache hits (below every answer and workflow)
- **[Review queue](https://megamiii.github.io/tieout/#/review):** accept or override the answers the agent sent to review, as a human reviewer would
- **[Evaluation](https://megamiii.github.io/tieout/#/eval):** accuracy, citation precision and abstention quality for the three configurations

| Trace of a recorded answer | Tie-out of 3M's FY2018 10-K |
|---|---|
| [![The trace waterfall: seven agent steps, each an LLM call and a tool call, all served from the recording](docs/trace.png)](https://megamiii.github.io/tieout/#/ask?q=3m-operating-margin-drivers) | [![Tie-out exceptions: two MD&A figures that differ from the XBRL line item, with the value-blind LLM's reasoning](docs/tieout.png)](https://megamiii.github.io/tieout/#/tieout?ticker=MMM&year=2018) |

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
- **Evaluation harness:** tested on the FinanceBench open-source benchmark plus numeric questions with ground truth from XBRL; CI replays every committed eval answer on each push
- **Observability:** a span-level trace of every agent run and workflow (LLM calls, tool calls, retrieval rankings, validation), with timings, tokens and cache hits, shown as a waterfall in the UI
- **MCP server:** the filing search and financial data tools are also exposed through the Model Context Protocol for AI assistants and IDEs that support it
- **Demo mode:** curated questions replay recorded agent runs, so anyone can try the public demo for free; running locally with your own free API key answers any question

## Architecture

```
        ┌─────────────────────────── Web UI (React + TypeScript) ────────────────────────────┐
        │ ask · tie-out · analytical review · trace waterfall · review queue · evaluation    │
        └──────────────────────────────────────────┬─────────────────────────────────────────┘
                                                   │ REST API (the static demo reads a JSON snapshot of it)
                     ┌─────────────────────────────▼────────────────────────────┐
                     │ FastAPI backend                                          │
                     │   Agent loop ──► LLM API (Gemini / Groq / Ollama)        │
                     │     ├─ search_filings ──► hybrid RAG                     │
                     │     ├─ read_section                                      │
                     │     ├─ get_financial_facts ──► XBRL facts                │
                     │     ├─ calculate                                         │
                     │     └─ submit_answer ──► validator (citations, figures)  │
                     │   Workflows: tie-out · analytical review                 │
                     │   Tracing · response cache = recordings · demo mode      │
                     └─────────────────────────────┬────────────────────────────┘
                                                   │
                     ┌─────────────────────────────▼────────────────────────────┐
                     │ Postgres + pgvector                                      │
                     │ filings · chunks · XBRL facts · runs · reviews · traces  │
                     └─────────────────────────────▲────────────────────────────┘
                                                   │
                     ┌─────────────────────────────┴────────────────────────────┐
                     │ MCP server (the same tools, for AI assistants and IDEs)  │
                     └──────────────────────────────────────────────────────────┘
```

**Design decisions**

The key choices, with what was measured (details per milestone below):

- **Numbers are computed, not generated:** every figure in an answer must appear in a cited source or a `calculate` result, or the validator rejects the answer and tells the model why. This costs coverage: the full agent answers 80.0% of FinanceBench questions, against 87.1% for the agent without the rule, and when the correct answers sent to review were inspected (validator v2), 6 of 10 contained arithmetic the model did in its head. In audit work an unverifiable figure is worse than a referral to a reviewer.
- **Text plus XBRL:** filing text explains, XBRL states exact values. Dense financial tables retrieve poorly (3M's FY2018 cash flow statement is not in the top 10 of any retriever for "capital expenditure"), while `get_financial_facts` returns the line item with a fact ID to cite. Adding XBRL and the calculator raised FinanceBench numeric accuracy from 78.3% to 87.0% and citation precision from 63.6% to 71.9%; the overall accuracy difference (84.3% → 85.7%) is within the noise of a 70-question set.
- **Hybrid retrieval, measured rather than assumed:** keyword search was expected to help with line-item names and exact figures, but on raw FinanceBench questions vector search alone retrieved better (hit@10 68.6% vs 54.3%), because questions and filings use different words. The agent writes its own queries and searches hybrid; a vector-only agent was not run, so whether hybrid helps the agent is untested.
- **Abstain instead of guessing:** an answer goes to the review queue when the model flags missing or conflicting evidence, its confidence is below 0.7, it fails validation twice, or the run hits its step or token limit. On FinanceBench, 35.7% of the answers sent to review were actually wrong, and half of all wrong answers were sent to review.
- **A demo that costs nothing and can't drift:** every LLM response is cached by request hash, so a run replays exactly. The public demo replays recorded runs as a static site, and CI replays all 330 committed eval answers on every push and fails if any changes.

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
- **One client, any OpenAI-compatible provider:** a small `httpx` client for `/chat/completions`, so `LLM_PROVIDER=groq|gemini|ollama|openai_compatible` switches providers. Groq's free-tier limits are built in; Gemini shows each project's free-tier limits in AI Studio instead of publishing them, so they are set in `.env` (`LLM_RPM`, `LLM_TPM`, `LLM_RPD`, `LLM_TPD`). Gemini returns its retry hint in the 429 body (`retryDelay`) rather than a `Retry-After` header, and the client reads both.
- **Model choice is driven by requests/day, not tokens:** in AI Studio (2026-10-09) this project's free tier allows `gemini-3.8-flash` 5 requests/min and **20 requests/day**, but `gemini-3.5-flash-lite` 15 requests/min, 250K tokens/min and **500 requests/day**. An agent question takes several requests, so 3.8 Flash would allow only a handful of questions a day; the default model is `gemini-3.5-flash-lite`.
- **Echo provider-specific fields back:** Gemini's thinking models attach `extra_content.google.thought_signature` to each tool call and reject the next request (HTTP 400) if it is not sent back. The client keeps every provider-specific field on a tool call and returns it verbatim, which fixes this without Gemini-specific code in the loop.
- **First live run** (recorded on 2026-10-09; the recording was retired in milestone 10 after a prompt change made it stale, and the demo now replays the same question from the eval run): "What was 3M's FY2018 capital expenditure in USD millions?" → $1,577 million, which matches FinanceBench's gold answer, citing the XBRL fact and the cash flow statement chunk (p. 60), validated and `answered` with confidence 1.00. It took 6 LLM requests and 29,150 tokens: the model found the XBRL value in step 1, then spent four more steps corroborating it in the text. Prompting it to stop once two sources agree is a candidate improvement, to be measured in milestone 5. It doesn't use an SDK, so retries, caching and token accounting are all visible in about 200 lines.
- **Response cache = recordings:** each request is serialized as canonical JSON (sorted keys, plus the provider URL) and hashed with SHA-256; the response is stored under that hash. Re-running an identical request costs no tokens. In offline mode the client serves only cached responses and raises on a miss, so a recorded run replays exactly, which is the mechanism for keyless tests and, later, the public demo.
- **Rate limiting before, retrying after:** a sliding-window limiter waits *before* a request that would exceed requests/min or tokens/min (estimated at ~4 characters per token). If a 429 still happens, the client retries after Groq's `retry-after`, using the same backoff code as the SEC downloader. A daily ledger (`data/usage/YYYY-MM-DD.json`) records tokens per model, cache hits, tokens saved and Groq's latest `x-ratelimit-*` headers; `tieout usage` reports it against the daily limits.
- **The loop is written by hand** (`backend/tieout/agent/loop.py`, no framework): send messages and tool schemas, execute each tool call, append each result as a `tool` message tied to its `tool_call_id`, and stop when the model answers in plain text.
- **Guardrails:** a maximum number of steps, where the last step forces a final answer (in milestone 3 with `tool_choice="none"`; since milestone 4 by forcing `submit_answer`) so the run doesn't end empty; a token budget per run, counted for cached responses too so a replay stops exactly where the live run did; and tool errors (invalid JSON, schema violations, unknown tickers) returned to the model as `{"error": ...}` so it can correct itself.

Tools and answer validation (milestone 4):

- **`get_financial_facts` answers what retrieval can't:** dense financial tables retrieve poorly, so line items come from XBRL. A phrase is matched to the company's own concepts by embeddings over concept label + description (2,807 concepts, embedded once and cached), keyword matches on the label, and a short alias list for common names ("capex" → `PaymentsToAcquirePropertyPlantAndEquipment`). For "3M capital expenditure FY2018" it returns 1,577,000,000 with its fact ID, the value the retrievers missed.
- **As originally reported:** each fiscal year's value comes from the earliest-filed 10-K that reports that period, which is what FinanceBench answers and an audit tie-out use. Quarterly durations and balance-sheet instants that are not fiscal-year ends are excluded.
- **Restatements are warnings, not abstentions:** 3,906 (company, concept, period) combinations have more than one reported value across 10-Ks, mostly later reclassifications. Routing all of them to review would flag almost everything, so a cited fact that was later reported differently adds a warning to the answer (shown to the reviewer) instead.
- **`read_section` reads one page at a time:** Item 8 can span 70 pages, far more than 8k tokens/min allows, so the tool returns one page's chunks with their IDs (for example the whole cash flow statement around a search hit).
- **`calculate` is a safe evaluator, not `eval`:** it walks a whitelisted Python AST (numbers, + − × ÷ // % **, unary minus, abs/round/min/max) with `Decimal` arithmetic, so `__import__('os')` can't run, 0.1 + 0.2 is exactly 0.3, and `9**9**9` is refused.
- **`submit_answer` is validated against the run's own evidence:** every tool result registers the sources it showed the model. A submission is rejected, with the specific problems sent back to the model, unless (1) every cited ID was returned by a tool in this run, which catches both invented IDs and real ones the model never saw; (2) every quote appears in its source (numeric quotes match at any scale); and (3) every figure in the answer appears in a cited source or a `calculate` result, allowing rounding and unit changes ($1,577 million ↔ 1,577,000,000; 14.9% ↔ 14.858). Rule 3 is how "never let the LLM do math" is enforced rather than requested. It is skipped in the ablation without a calculator, which could not satisfy it.
- **Abstention:** the answer is `needs_review` if the model flags it (missing, ambiguous or conflicting evidence), its confidence is below 0.7, it fails validation twice, or the run hits its step or token limit without a valid submission. The reasons are kept for the review queue. On the last step `tool_choice` forces `submit_answer`, so running out of steps still produces a validated answer or a clear review reason. A plain-text reply is answered with a request to call `submit_answer`.
- **Replay without a key, database or model:** outputs of the database tools are recorded by a hash of (tool, arguments), like LLM responses. Since validation only needs the run's evidence, a recorded run replays fully offline, which is what the CI eval and the public demo need.

Web UI and review queue (milestone 6):

- **Runs are persisted, reviews are append-only:** every answered question is stored in `agent_runs` (answer, status, review reasons, citations, usage, trace). Reviewer decisions go to `review_decisions` and are never updated in place: the latest decision is the current state and the history shows who accepted or overrode what, and when, which is what an audit trail needs.
- **A queue with real cases from day one:** `tieout runs import-evals agent_full` loads the agent's eval results into the queue without any LLM calls, so the review queue opens with the actual FinanceBench questions the agent sent to review, with their real reasons (for example "the number $9,983 is not in any cited source or calculate result": the model summed the quick-ratio components itself instead of using the calculator). Only agent runs are imported; the baseline exists for comparison, not review.
- **Highlighting is computed on the server and unit-tested:** the API returns each cited passage as segments with the quoted spans marked. Matching tolerates PDF line breaks inside table rows and quotes with or without thousands separators ("1577" matches "1,577" but not the "15,77" inside "15,770"); when the quote as a whole isn't in the passage, its figures are highlighted instead and the UI says so. Several quotes from the same source are merged into one card, and each passage scrolls to its first highlight.
- **Model output is rendered as Markdown, safely:** the model writes Markdown; the UI renders it with `react-markdown`, which never renders raw HTML. The prompt was not changed mid-eval to avoid Markdown, since that would mix two agent versions within one experiment.
- **Frontend stack:** React 19 + TypeScript (strict) with Vite, ESLint (`typescript-eslint` strict) and plain CSS with light and dark themes; three views (ask, review queue, eval dashboard) with a tiny hash router instead of a router dependency. Vite is pinned to 6 because Vite 8 requires Node ≥ 20.19.

Tie-out and analytical review (milestone 7):

- **Mostly deterministic, LLM only where text must be interpreted:** claim extraction, matching, material-change selection and quote verification are code with unit tests; the LLM is used for one judgment per item. A tie-out of a whole 10-K costs 3–24 LLM requests and an analytical review 8, which fits the free tier.
- **Claims keep their written precision:** only dollar amounts with an explicit scale in MD&A prose are claims ("$1.6 billion", "$1,577 million"). "$1.6 billion" may be anything from 1.55 to 1.65 billion, while "$1,577 million" must be within half a million, so matching allows exactly the rounding the author used. Each claim also records the fiscal year it refers to ("$21.3 billion in fiscal 2023"; "…for 2022, 2021 and 2020, respectively") and whether it is quarterly or a non-GAAP / adjusted figure. Quarterly amounts and amounts rounded too coarsely to tie out (where the rounding exceeds 5%, as in "$0.1 billion") are reported as out of scope.
- **A match needs value, year and meaning:** a fact matches if it equals the claim within its rounding, for that fiscal year, and its concept ranks in the top 15 of the filing's concepts by embedding similarity to the sentence. Value alone is not enough: "$8 million" lies within rounding of several unrelated facts. In a hand check of 20 random matches, 19 were the right line item and year (one was a year-assignment coincidence).
- **Exceptions are identified by a value-blind LLM, and code compares the numbers:** when no related fact has the value but a closely related concept has a close one (within ±20%), the LLM is shown the sentence with the figure replaced by `[AMOUNT]` and the candidate line items *without values*, and names the line item [AMOUNT] reports (same measure, scope and period) or none. Code then compares the values. The first version showed the LLM both numbers, and it reasoned "the text says $2.523 billion whereas XBRL says $2.803 billion, so these are different quantities": a different value was taken as evidence of a different quantity, so real discrepancies could never be flagged. Without any LLM check, embedding similarity raised 79 candidate exceptions in five filings, and a spot check of 25 found almost all were different quantities ("Health Care segment sales" vs total SG&A).
- **Detection is measured with planted errors (mutation testing):** audited 10-Ks rarely contain real discrepancies, so "no exceptions found" says nothing about whether the check works. 40 claims that tie out were altered by 5–10% (beyond their rounding) and run through the tie-out: the value-aware adjudicator caught **2 of 40**, the value-blind one **30 of 40**, and **28 of 40 (70%)** after adjusted / non-GAAP figures were excluded from exceptions by design (two planted errors sat in "adjusted" sentences). None was flagged against the wrong fact. Report: [evals/reports/tieout_mutation.json](evals/reports/tieout_mutation.json).
- **What reviewers see on real filings:** on five 10-Ks (3M FY2018, Nike FY2023, AMD FY2022, Boeing FY2022, General Mills FY2022), 751 claims: 139 matched, 11 exceptions, 474 not in XBRL (segment figures, non-GAAP measures, amounts not tagged in the statements) and 127 out of scope. All exceptions were checked by hand (14 before the non-GAAP rule) and none was a real discrepancy: they were segment vs company totals ("BDS loss from operations" vs Boeing's total), net vs gross repurchases, a component vs its line item (accrued trade promotions vs accrued marketing), balance-sheet vs cash-flow changes, cumulative program vs annual amounts, and a few year-assignment errors in truncated sentences. So at 70% detection of planted errors, a reviewer currently clears about two false exceptions per filing; exceptions are review items, not findings.
- **Analytical review:** line items that changed at least 20% year over year and are at least 1% of revenue (total assets if there is no revenue concept), largest dollar changes first, at most 8. For each, MD&A (Item 7) is searched with the line item's plain name plus its two figures, which MD&A sentences usually restate, and the LLM must quote management's explanation from those passages; the quote is verified against the cited chunk. On 3M FY2018, Nike FY2023 and AMD FY2022 the first version explained 4 of 24 changes: it picked redundant totals (one acquisition moved AMD's assets, liabilities-and-equity, equity and paid-in capital) and searched the whole filing with XBRL label jargon. Skipping balance-sheet identities and equity / comprehensive-income components, merging identical movements and searching MD&A with the figures raised this to **11 of 24**; by hand, 10 of the 11 quotes do explain the change (one quotes the definition of goodwill rather than why it changed). "No explanation found" is itself an audit observation, but some of the 13 are retrieval misses (AMD's cost of sales).

Tracing (milestone 8):

- **A span tree, not a log:** each agent run or workflow records a tree of spans: `agent` → `step` → `llm` / `tool` → `retrieval`, plus `validation` (with the problems sent back to the model), `adjudication` (tie-out) and `explain` (analytical review). Each span has its start offset, duration, input, output, error, tokens and whether it was served from the cache. The UI shows it as a waterfall under every run in the review queue and every tie-out or analytical review; clicking a span shows its input, output and metadata.
- **No tracing library, and no changes to function signatures:** the active tracer lives in a Python `contextvar`, so the LLM client, the tools, search and the validator call `with span(...)` without a tracer being passed through the call chain. With no active tracer (unit tests, evals), `span` is a no-op. FastAPI runs each request in its own context, so concurrent requests never write to the same trace. The tracer and its store are about 250 lines (`backend/tieout/tracing/`).
- **LLM spans store only what is new:** the agent resends the whole conversation on every step, so storing each request in full would repeat the system prompt and every earlier tool result. Each `llm` span stores only the messages added since the previous call, plus the offset in the conversation; the UI says how many earlier messages it doesn't repeat. On the AMD quick-ratio run (8 LLM calls) this stores 22K characters instead of 126K (5.7× smaller). The full request is still recoverable from the response cache by the request hash kept in the span's metadata.
- **Retrieval spans show why a passage was found:** each hit records its vector rank, keyword rank and fused RRF score, and the span records how many candidates each retriever returned, so a reviewer can see whether a citation came from semantic or keyword search.
- **Cost is reported in free-tier units:** the trace totals count LLM requests and tokens only for calls that reached the provider; cached calls are counted separately. The project uses only free tiers, so the dollar cost is always $0 and the meaningful budget is requests and tokens per day (see `tieout usage`).
- **Saved with the run, in the same transaction:** a trace (`traces` + `spans` tables) is written in the same database transaction as the agent run or workflow it belongs to, so there is never a run without its trace or the reverse. `tieout ask --save` and `tieout workflow … --save` store traces from the CLI too.
- **Replays show what was cached:** a replayed run marks every LLM span `cached`, so the trace of a recorded demo answer reads "0 LLM requests (+8 from cache)". This also exposed a stale recording: the first live run (milestone 3) no longer replayed because the system prompt changed after it was recorded, so the request hashes differ. Milestone 10 replaced it.

MCP server (milestone 9):

- **The same tools, not a copy:** the server lists the agent's own `Tool` objects with the JSON schemas the agent's LLM sees, and runs calls through the same validate-then-run function as the agent loop. An MCP client gets the same argument checks and the same error messages ("unknown ticker 'XYZ'; supported: [...]"), and a change to a tool changes both. A test checks that the listed schemas are identical.
- **Official SDK, low-level server:** the MCP Python SDK's high-level decorator API builds schemas from function signatures, which would define every tool a second time. The low-level `Server` takes the existing schemas and handlers directly.
- **What's exposed:** `search_filings`, `read_section`, `get_financial_facts` and `calculate`. `submit_answer` is not, because it is only meaningful inside the agent loop, where it is validated against the evidence the run has seen.
- **Errors the model can act on:** a bad call to a real tool (invalid arguments, an unsupported company, a disallowed calculator expression) returns a tool result with `isError` and the message, so the client's model can retry, as the MCP spec recommends. Only calling a tool that doesn't exist is a protocol error.
- **Coverage in the server instructions:** the instructions sent at connection list the 13 companies and their fiscal years, so the assistant knows what it can ask about before spending a call.
- **Costs nothing to run:** there is no LLM inside the server, so it uses no API quota. It runs over stdio only, for local clients; serving it over HTTP from the deployed demo would be a separate step.
- **Tested in-process and over stdio:** unit tests use the SDK's in-process client (tool listing, schemas, structured results, error results). An end-to-end check launched `tieout mcp` as a subprocess against the local database: the client connected in 2.9 s, `get_financial_facts` returned 3M's FY2018 capital expenditure ($1,577 million, with its fact ID) in 1.6 s on the first call (the embedding model loads lazily), and `search_filings` answered in 0.1 s. Logs go to stderr, since stdout carries the protocol.

Demo mode and deployment (milestone 10):

- **One replay mechanism, no special cases:** with `DEMO_MODE=true` the LLM client and the database tools run offline against `recordings/demo/`. A recorded question replays exactly; anything else misses the recording and gets a clear message ("this public demo only replays recorded runs…") instead of an error. There is no list of allowed questions in the code: the curated questions are simply the ones that were recorded, and the container has no API key at all.
- **Recordings are assembled from traces, at zero cost:** a trace names every recording file its run used (the request hash on LLM spans and, since this milestone, the recording hash on tool spans). `tieout demo build` replays six curated eval questions from the eval recordings and the eight stored workflows from the local LLM cache, copies exactly the files their traces name into `recordings/demo/` (776 KB, 166 files), and then replays each one from that directory alone to prove the copy is complete. Building the demo made no LLM requests, and the rebuilt workflows reproduce the published results exactly (751 claims, 139 matched, 11 exceptions; 11 of 24 changes explained). The curated list, with what each question demonstrates, is `config/demo_questions.json`.
- **A public database that doesn't grow:** a replayed question is stored once (as a `demo` run) and served from the database afterwards, so repeated clicks return the same run in milliseconds. Workflows are served from the runs stored with the database, and an unrecorded filing is refused before any work: the first version ran 7.3 s of claim extraction before its first LLM call missed the recording; refusing up front takes under 0.2 s. Review decisions, the one remaining public write, are limited to 30 per client per hour.
- **Hash inputs are pinned in the image:** a request hash covers the provider URL and the model, so the Dockerfile sets `LLM_PROVIDER=gemini` and `LLM_MODEL=gemini-3.5-flash-lite`, the values the recordings were made with. Changing either would turn every replay into a cache miss.
- **Hosting moved to GitHub Pages, because Hugging Face's free tier no longer covers Docker:** the first plan was a Docker Space on Hugging Face with the database on Neon. Before deploying, Hugging Face's documentation (checked 2026-10-13) said Docker and Gradio Spaces now require a paid plan to create, which this project can't use. The demo only replays recorded runs, so everything it can show is fixed data: `tieout demo export-static` calls the real demo API endpoints in-process (FastAPI's test client, in demo mode, inside a transaction that is rolled back) and writes each response to `frontend/demo-data/` (1.7 MB: 23 runs, 8 workflows, 15 traces, the eval report). The frontend's static build (`vite build --mode static`) answers its requests from those files, so the page shows exactly what the demo server would, with no server, database or API key, and a GitHub Actions workflow publishes it to GitHub Pages on every push. The one thing a static site can't do is share writes: review decisions are saved in the visitor's browser (localStorage), and the demo banner says so.
- **Self-hosting stays possible:** the multi-stage Dockerfile builds the UI with Node, then installs the API on `python:3.12-slim` with the demo recordings, the eval reports and the quantized embedding model (65 MB) baked in, and serves both on port 7860 as a non-root user (855 MB image, no API key; `.dockerignore` keeps `.env`, `data/` and the eval recordings out). `deploy/seed-neon.sh` copies the local database into any Postgres, such as a free Neon project, using `pg_dump | psql` run inside the local Postgres container; a fresh copy is 160 MB. Neon hands out `postgresql://` URLs, which SQLAlchemy would map to the psycopg2 driver, so the engine rewrites them to psycopg 3.
- **Tested locally, without a server for the static site:** the Docker image ran in demo mode against the local database with no API key: recorded questions replayed in 0.1 s with "0 LLM requests (+7 from cache)" in the trace, unrecorded questions and filings were refused, and every page fit a 375 px phone screen (the nav tabs now scroll instead of widening the page). The static build was then served under `/tieout/`, as on GitHub Pages, with the API and the container stopped: recorded questions, tie-outs, traces and the eval dashboard loaded, unrecorded ones got the same refusal as the server, and an accepted review moved to the Reviewed queue and survived a reload. The seed script was tested by restoring into a scratch database, twice, to check that re-seeding replaces rather than fails.

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

After `tieout data index`: 705 sections and 13,003 chunks (bge-small-en-v1.5, 384 dimensions). The local Postgres database is 254 MB (chunks 131 MB including HNSW and GIN indexes, XBRL facts 102 MB, sections 12 MB); a fresh copy without the update history is 160 MB.

## Evaluation

Results for the full agent (agent + hybrid RAG + XBRL + calculator) on `gemini-3.5-flash-lite` (Gemini free tier), measured 2026-10-09 to 2026-10-13. Full report: [evals/reports/eval.md](evals/reports/eval.md).

| Metric | Description | Result |
|---|---|---|
| FinanceBench accuracy | Correct answers on the 70 open-source FinanceBench questions about the supported 10-Ks (numeric tolerance or an LLM judge) | **85.7%** (60/70; 95% CI 75.7–92.1%) vs 44.3% for the baseline |
| XBRL numeric accuracy | 40 generated questions with exact ground truth, graded within 0.5% (lookups) or 1% (growth rates) | **90.0%** (36/40; 95% CI 76.9–96.0%) vs 55.0% for the baseline |
| Citation precision | Share of citations that point at the gold evidence (definition below) | 71.9% on FinanceBench (196 citations), 55.6% on XBRL (63) |
| Abstention quality | Answered rate, and precision / recall of "needs review" | 80.0% answered on FinanceBench; of answers sent to review, 35.7% were actually wrong (precision); of wrong answers, 50.0% were sent to review (recall) |
| Usage per question | LLM requests and tokens (the free tier has no dollar cost) | 6.5 requests, 31.8k tokens on FinanceBench; 5.0 requests, 19.6k tokens on XBRL; $0 |
| Latency | p50 / p95 per question | 8.3 s / 28.9 s on FinanceBench, 4.5 s / 17.7 s on XBRL (from the live run; see note) |

**Experiments**

| Configuration | FinanceBench accuracy | XBRL numeric accuracy | Requests / tokens per question (FinanceBench) |
|---|---|---|---|
| Single prompt, vector RAG only (baseline) | 44.3% | 55.0% | 1.0 / 2.7k |
| Agent + hybrid RAG | 84.3% | 87.5% | 6.4 / 32.9k |
| Agent + hybrid RAG + XBRL + calculator | 85.7% | 90.0% | 6.5 / 31.8k |

All three configurations use the same model, citation rules, confidence threshold (0.7) and graders; each question was run once at temperature 0.

**What the results say**

- **The agent loop is where the gain comes from.** Going from one prompt over retrieved chunks to an agent that searches iteratively, reads whole pages and must pass citation checks raises FinanceBench accuracy from 44.3% to 84.3%. The agent got 31 questions right that the baseline missed and 3 wrong that it got right (exact McNemar test, p < 10⁻⁶); on the XBRL set, 14 vs 1 (p ≈ 0.001).
- **XBRL + calculator help where expected, but the overall accuracy difference is within noise.** They raise FinanceBench numeric accuracy (78.3% → 87.0%), citation precision (63.6% → 71.9% on FinanceBench, 42.0% → 55.6% on XBRL) and XBRL accuracy when answered (89.7% → 94.7%). Overall accuracy moves 84.3% → 85.7% on FinanceBench and 87.5% → 90.0% on XBRL, but the two agents disagree on only 9 and 5 questions (p = 1.0 for both): with 70 and 40 questions, the confidence intervals are about ±8–10 points, so this eval cannot tell these two configurations apart on accuracy alone.
- **The strict "no unverified arithmetic" rule trades coverage for verifiability.** The full agent answers 80.0% of FinanceBench questions versus 87.1% for the agent without a calculator (which has no figure check), and most of what it sends to review is actually correct (abstention precision 35.7%): the model summed or averaged figures itself instead of calling `calculate`, and the validator correctly refused to accept unverified arithmetic. In audit work that is the intended behaviour, but it shows up as lower coverage. Ways to recover coverage without dropping the rule (prompting the model to compute intermediate values with `calculate`, or letting the validator re-derive simple sums of cited figures) are open next steps.
- **The baseline abstains precisely but often.** It answers only 50% of FinanceBench questions, and 91% of what it declines it would have gotten wrong: one prompt over eight vector-retrieved chunks frequently lacks the table it needs and says so.
- **Cost:** the agent uses about 6.5 LLM requests and 12× the tokens of the baseline per question. On the free tier that means requests per day, not money, is the limit: the full eval took four quota days.
- **Caveats:** one run per question (no variance across seeds), small sets, an LLM judge that is lenient on figures inside text answers (see the spot check below), and a numeric set whose ground truth comes from XBRL, which favours the XBRL tool by construction.
- **Latency note:** the full agent's final run replayed its recorded steps (after the validator fixes below), and replays are excluded from latency, so its latency is taken from its live run (the archived v1 row in the report: same model and the same step sequence up to the validator change).

### How the eval works (milestone 5)

Status: complete (all 110 questions graded for each configuration). The free tier allows 500 agent requests a day, so the runs were spread over four quota days.

- **Set A, FinanceBench:** the 70 open-source FinanceBench questions about the supported 10-Ks (23 with a numeric gold answer, 47 text). Gold evidence pages and the figures in the gold evidence passages are kept for citation scoring.
- **Set B, XBRL numeric:** 40 questions generated from XBRL with exact `Decimal` ground truth: 20 single-value lookups ("Nike's total revenue in fiscal year 2023, in USD millions") and 20 year-over-year % changes (which need the calculator), sampled round-robin across companies (2–4 each). Growth questions are only generated where the prior year was never restated, so there is one right answer. The ground truth comes from XBRL, so this set favours the XBRL tool by construction.
- **Graders:** numeric answers pass if any figure in the answer equals the gold value within 1% (0.5% for lookups) or the figure's rounding, at any scale ($1,577 million ↔ 1,577,000,000; 8.7% ↔ 0.087). Any figure may match, which is lenient by design. Text answers are graded by an LLM judge that sees the question, the gold answer and FinanceBench's justification and returns a verdict with its reasoning (stored per item). The judge is `gemini-3.1-flash-lite`, a different model from the agent (`gemini-3.5-flash-lite`) to limit self-preference bias, with its own daily quota. A spot check of 8 judge verdicts found all 4 "incorrect" and 3 of 4 "correct" verdicts clearly right; one "correct" verdict accepted a different debt figure because the conclusion matched, so the judge is lenient on figures inside text answers.
- **Metrics:** accuracy counts abstentions and errors as not correct; *accuracy when answered* and *answered rate* are reported alongside, since abstaining more can raise one and lower the other. **Citation precision** is the share of citations that point at the gold evidence: a gold XBRL fact, a chunk on a gold evidence page of the right filing, a source stating the gold value, or a source stating a distinctive figure (≥ 100, not a year) that also appears in FinanceBench's gold evidence, equal up to its rounding. It is computed at report time from what each citation states, so the rule can be improved without re-running the agent. **Abstention precision** is P(the answer was wrong | it went to review), and **abstention recall** is P(it went to review | the answer was wrong); answers sent to review are still graded so both can be measured. **Usage** is LLM requests and tokens per question (the free tier has no dollar cost; requests/day is the binding limit). **Latency** p50/p95 is measured on live runs (LLM time + tool time); replays from the cache are excluded.
- **Experiments:** (1) baseline: one LLM call over the top-8 vector-search chunks, with company and fiscal year taken from the question by string matching, asking for JSON with citations; (2) agent with `search_filings` (hybrid), `read_section` and `submit_answer` only; (3) the full agent with `get_financial_facts` and `calculate` as well. All three use the same model, prompt rules for citations, confidence threshold (0.7) and graders.
- **Resumable and replayable:** one result file per (experiment, question) in `evals/runs/`; re-running skips finished items, and stops before the daily request quota (counted on the provider's quota day: midnight Pacific for Gemini). LLM calls, tool outputs and judge calls are recorded in `recordings/evals/`, so the whole eval replays offline.

**What the eval found before any headline number** (each fixed, with a regression test):

- The validator parsed "fiscal year 2015," as the figure `2015,` and rejected correct answers; numbers now require a thousands separator to be followed by exactly three digits.
- In the first agent pass, 21 of 32 FinanceBench runs ended without a valid answer (12 out of steps, 9 over a 30k-token budget): the model first submitted on the last step, so any rejection ended the run. One rejection was a fabricated quote ("3,541" where the source says 4,134), which the validator correctly caught. `submit_answer` is now forced on the last two of eight steps, the run budget is 60k tokens (requests, not tokens, are Gemini's binding limit), and the schema states that quotes must be under 300 characters and that calculate results are not citable. In a 5-question re-run, all 5 ended with a validated answer or an explicit review decision.
- The first citation-precision rule (gold page or gold value) scored correct XBRL citations as unsupported on text questions (22%); a looser second rule credited coincidences ("31", figures 0.5% apart) and inflated the baseline from 47.8% to 89.6%. The current rule is described above, and both failure modes have tests.

**Validator changes measured during the eval** (the full agent was re-run after each; recorded steps replay from the cache, so each re-run cost a handful of requests):

| Validator | FinanceBench answered | accuracy | abstention precision / recall | What changed |
|---|---|---|---|---|
| v1 (archived) | 75.7% | 85.7% | 35.3% / 60.0% | — |
| v2 | 77.1% | 85.7% | 37.5% / 60.0% | figures that appear in the question count as supported ("more than 20% of revenue?") |
| current | 80.0% | 85.7% | 35.7% / 50.0% | dates and references ("December 31", "Note 21") are not figures |

The v2 hypothesis (that question figures caused most of the over-abstention) was wrong: it changed one answer. Inspecting the 10 correct answers still sent to review showed 6 were unverified in-head arithmetic (the rule working as intended), 2 were date/reference false positives (fixed in the current validator) and 2 were quotes that don't match their source exactly.

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
- Optional: a free API key from [Groq](https://console.groq.com) or [Google AI Studio](https://aistudio.google.com/apikey) (Gemini), or [Ollama](https://ollama.com) for local runs. None is needed for demo mode, and the project never needs a paid API key.

### Run locally

```bash
git clone https://github.com/megamiii/tieout.git
cd tieout
cp .env.example .env            # set SEC_USER_AGENT to "<name> <your email>"
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
docker compose up -d --wait     # Postgres + pgvector
tieout data all                 # select, download (cached), load XBRL, concept catalog, parse + chunk + embed + index
tieout search "capital expenditures" --ticker MMM --year 2018
tieout ask "What was 3M's FY2018 capital expenditure?"   # needs a free Gemini or Groq key, or Ollama
tieout ask "..." --recording recordings/my-run            # record LLM + tool calls
tieout ask "..." --recording recordings/my-run --offline  # replay: no key, DB or model
tieout usage                    # today's tokens vs the free-tier daily limits
tieout workflow tieout MMM 2018 # tie-out MD&A figures against XBRL
tieout workflow review MMM 2018 # analytical review of material changes
tieout eval tieout              # mutation test: planted errors in tied-out claims
pytest -q
```

Web UI (two terminals; open http://localhost:5173):

```bash
tieout serve --reload           # API on :8000  (make api)
npm --prefix frontend install && npm --prefix frontend run dev   # UI on :5173 (make web)
tieout runs import-evals agent_full   # optional: fill the review queue from eval results
```

The first `tieout data all` embeds 13,003 chunks on the CPU (about 70 minutes on a 15 W laptop CPU); later runs reuse the embedding cache and finish in a few minutes. `make ingest`, `make test` and `make lint` wrap the same commands. See [Build progress](#build-progress).

### Publish the demo (GitHub Pages)

The public demo is a static site: no server, database, API key or other account.

1. In the GitHub repository, open **Settings → Pages** and set **Source** to **GitHub Actions**.
2. Push to `main`. The [Pages workflow](.github/workflows/pages.yml) builds the frontend in static mode with `frontend/demo-data/` and publishes it at `https://<user>.github.io/<repo>/`.

After changing the curated questions (`config/demo_questions.json`) or the stored workflows, rebuild the recordings and the snapshot, then commit both:

```bash
tieout demo build            # recordings/demo from eval runs and cached workflows (no LLM requests)
tieout demo export-static    # frontend/demo-data from the demo API
```

To preview the static site locally: `npm --prefix frontend run build:static`, then `npm --prefix frontend run preview`.

### Self-host the demo server (optional)

```bash
docker build -t tieout:demo .
docker run -p 7860:7860 -e DATABASE_URL=postgresql://tieout:tieout@host.docker.internal:5432/tieout tieout:demo
```

Open http://localhost:7860. To run it elsewhere, copy the database to any Postgres (for example a free [Neon](https://neon.tech) project) and pass its URL as `DATABASE_URL`:

```bash
NEON_DATABASE_URL='postgresql://USER:PASSWORD@HOST/neondb?sslmode=require' sh deploy/seed-neon.sh
```

### Evaluate

```bash
tieout eval retrieval           # retrieval hit@k / MRR -> evals/reports/retrieval.{md,json}
tieout eval run agent_full      # answer + grade (resumable; needs a free API key)
tieout eval report              # evals/reports/eval.{md,json}
tieout eval verify              # replay every committed answer offline; fails if any differs
```

### Tests and CI

`pytest -q` runs 204 tests; the 20 database tests need Postgres (`docker compose up -d`) and are skipped, after one 3-second connection check, when it isn't running. [CI](.github/workflows/ci.yml) runs on every push and pull request, with no API key:

- **Backend:** ruff, mypy (strict) and the full test suite against a fresh pgvector Postgres service, so the database tests run.
- **Eval replay:** `tieout eval verify` re-answers all 330 committed eval items (3 configurations × 110 questions) from `recordings/evals/` with no key, database or embedding model, and fails if any answer, status or citation differs from `evals/runs/` or needs a request that was never recorded. Locally this takes about 8 seconds. It also confirmed that the validator fix committed after the full agent's final run changed none of its answers.
- **Frontend:** ESLint, TypeScript, and both builds (the app and the static demo).

### Use the MCP server

The filing search and XBRL tools are available to any MCP client. The server runs locally over stdio and reads the local Postgres database, so start the database first (`docker compose up -d`) and load the data. No API key is needed: the client's own model does the reasoning.

```bash
tieout mcp
```

Most MCP clients are configured with a JSON entry like this one; use the absolute path of the `tieout` executable in your virtual environment, then restart the client:

```json
{
  "mcpServers": {
    "tieout": {
      "command": "/absolute/path/to/tieout/.venv/bin/tieout",
      "args": ["mcp"]
    }
  }
}
```

On Windows the command is `C:\\path\\to\\tieout\\.venv\\Scripts\\tieout.exe` (backslashes are escaped in JSON). Paths inside the project (`.env`, `data/`, `config/`) are resolved from the package location, so the client can start the server from any working directory.

Tools: `search_filings`, `read_section`, `get_financial_facts`, `calculate`. The server's instructions list the companies and fiscal years that are indexed.

## Build progress

- [x] 1. Data + caching: FinanceBench sample, company selection, EDGAR filings + XBRL facts, cached downloads, Postgres load
- [x] 2. Parsing (10-K sections with page numbers) + hybrid retrieval, with a retrieval eval
- [x] 3. Agent loop with one tool
- [x] 4. All tools + answer validation (first live run on Gemini)
- [x] 5. Evals
- [x] 6. UI + review queue
- [x] 7. Tie-out + analytical review
- [x] 8. Tracing
- [x] 9. MCP server
- [x] 10. Demo mode + deployment (static demo on GitHub Pages)
- [x] 11. CI + README polish

## Project structure

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
│   ├── agent/            # agent loop, tools, calculator, answer validation, recording
│   ├── evals/            # eval sets, graders, runner, metrics, retrieval eval
│   ├── api/              # FastAPI app, demo mode
│   ├── tracing/          # span tree per run (contextvar tracer), stored in Postgres
│   ├── mcp_server/       # MCP server over stdio (the agent's data tools)
│   ├── workflows/        # tie-out (claims, matching, adjudication), analytical review
│   ├── runs.py           # persisted runs, review decisions, citation views
│   ├── highlight.py      # quote highlighting in cited passages
│   ├── ingest.py         # select → download → load pipeline
│   ├── indexing.py       # parse → chunk → embed (cached) → index pipeline
│   ├── demo_recordings.py # assemble recordings/demo from traces
│   ├── demo_static.py    # export the demo API as JSON for the static site
│   └── cli.py            # `tieout` command line
├── frontend/             # React + TypeScript UI (Vite); demo-data/ is the static demo snapshot
├── config/               # selected companies and filings, curated demo questions
├── evals/                # eval sets, per-question results, reports
├── recordings/           # recorded LLM + tool outputs: eval runs and the public demo
├── deploy/               # copy the database to a hosted Postgres (self-hosting)
├── tests/                # unit tests + DB tests (skipped without Postgres)
├── docs/                 # README screenshots, taken from the static demo
├── Dockerfile            # one image for self-hosting the demo server
├── .github/workflows/    # CI, and the GitHub Pages deployment of the static demo
└── docker-compose.yml    # Postgres 16 + pgvector
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
