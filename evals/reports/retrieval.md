# Retrieval eval

Questions: 70 FinanceBench questions on supported 10-Ks. A hit means a top-k chunk is on a gold evidence page of the right filing. All configurations filter to the question's filing unless noted.

| Configuration | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| vector | 20.0% | 42.9% | 54.3% | 68.6% | 0.343 | 46 | 87 |
| keyword, ts_rank_cd (no IDF) | 4.3% | 10.0% | 15.7% | 25.7% | 0.093 | 56 | 182 |
| keyword, BM25-lite IDF | 10.0% | 17.1% | 17.1% | 24.3% | 0.138 | 82 | 130 |
| hybrid RRF, ts_rank_cd keyword | 12.9% | 27.1% | 32.9% | 50.0% | 0.227 | 109 | 268 |
| hybrid RRF, BM25-lite keyword | 11.4% | 28.6% | 37.1% | 54.3% | 0.234 | 121 | 217 |
| hybrid RRF, BM25-lite, no filing filter | 5.7% | 22.9% | 28.6% | 42.9% | 0.159 | 472 | 730 |

Run metadata: date=2026-10-07, embedding_model=BAAI/bge-small-en-v1.5, chunk_max_tokens=400, chunk_overlap_tokens=50, chunks_indexed=13003, rrf_k=60
