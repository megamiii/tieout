# Eval report

Run metadata: date=2026-10-10, agent_model=gemini-3.5-flash-lite, judge_model=gemini-3.1-flash-lite, provider=gemini

## financebench

| Configuration | graded | accuracy | answered | acc. when answered | numeric | text | citation precision | abstention P / R | requests/q | tokens/q | latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single prompt, vector RAG only (baseline) | 70/70 | 44.3% | 50.0% | 80.0% | 43.5% | 44.7% | 59.7% (67) | 91.4% / 82.1% | 1.0 | 2,702 | 1.3 s / 11.6 s |
| Agent + hybrid RAG | 0/70 | - | - | - | - | - | - (0) | - / - | - | - | - s / - s |
| Agent + hybrid RAG + XBRL + calculator | 70/70 | 85.7% | 75.7% | 92.5% | 87.0% | 85.1% | 71.1% (197) | 35.3% / 60.0% | 6.6 | 32,261 | 8.3 s / 28.9 s |

## xbrl_numeric

| Configuration | graded | accuracy | answered | acc. when answered | numeric | text | citation precision | abstention P / R | requests/q | tokens/q | latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single prompt, vector RAG only (baseline) | 40/40 | 55.0% | 72.5% | 75.9% | 55.0% | - | 54.5% (33) | 100.0% / 61.1% | 1.0 | 3,044 | 1.1 s / 9.6 s |
| Agent + hybrid RAG | 0/40 | - | - | - | - | - | - (0) | - / - | - | - | - s / - s |
| Agent + hybrid RAG + XBRL + calculator | 39/40 | 89.7% | 94.9% | 94.6% | 89.7% | - | 55.7% (61) | 100.0% / 50.0% | 5.0 | 19,595 | 4.5 s / 17.7 s |
