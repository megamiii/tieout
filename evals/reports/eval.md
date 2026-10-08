# Eval report

Run metadata: date=2026-10-09, agent_model=gemini-3.5-flash-lite, judge_model=gemini-3.1-flash-lite, provider=gemini

## financebench

| Configuration | graded | accuracy | answered | acc. when answered | numeric | text | citation precision | abstention P / R | requests/q | tokens/q | latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single prompt, vector RAG only (baseline) | 70/70 | 44.3% | 50.0% | 80.0% | 43.5% | 44.7% | 47.8% (67) | 91.4% / 82.1% | 1.0 | 2,702 | 1.3 s / 11.6 s |
| Agent + hybrid RAG | 0/70 | - | - | - | - | - | - (0) | - / - | - | - | - s / - s |
| Agent + hybrid RAG + XBRL + calculator | 24/70 | 79.2% | 75.0% | 88.9% | - | 79.2% | 22.0% (82) | 50.0% / 60.0% | 6.9 | 34,350 | 16.5 s / 41.4 s |

## xbrl_numeric

| Configuration | graded | accuracy | answered | acc. when answered | numeric | text | citation precision | abstention P / R | requests/q | tokens/q | latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single prompt, vector RAG only (baseline) | 40/40 | 55.0% | 72.5% | 75.9% | 55.0% | - | 54.5% (33) | 100.0% / 61.1% | 1.0 | 3,044 | 1.1 s / 9.6 s |
| Agent + hybrid RAG | 0/40 | - | - | - | - | - | - (0) | - / - | - | - | - s / - s |
| Agent + hybrid RAG + XBRL + calculator | 0/40 | - | - | - | - | - | - (0) | - / - | - | - | - s / - s |
