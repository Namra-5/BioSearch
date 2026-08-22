# BioSearch AI — Evaluation Findings

**Run date:** 2026-08-22T11:13:46.579261+00:00
**Total runtime:** 24.1s
**Queries:** 5

## Executive Summary

TF-IDF and BioBERT showed no statistically significant difference in mean GCS at this sample size (179.0 vs 215.6; exact sign-test p=0.38, n=5; 95% bootstrap CI for the mean delta: -22.0 to 85.6, spanning zero). The per-query pattern suggests that BioBERT may perform relatively better on concept/synonym-heavy queries, while TF-IDF may perform relatively better on exact-terminology queries, but this directional pattern cannot be established statistically with only 5 queries. BioBERT's one-time model-load cost was 23.0s (first query, Q1); steady-state warm-query latency over the remaining 4 queries averaged 0.203s, approximately 7.8× slower than TF-IDF's 0.026s per query. The blended per-query mean of 4.8s reported below includes the one-time load cost and should not be read as steady-state latency.

## Key Findings

1. **Primary metric (GCS):** TF-IDF and BioBERT showed no statistically significant difference in mean GCS at this sample size (179.0 vs 215.6; exact sign-test p=0.38, n=5; 95% bootstrap CI for the mean delta: -22.0 to 85.6, spanning zero). The per-query pattern suggests that BioBERT may perform relatively better on concept/synonym-heavy queries, while TF-IDF may perform relatively better on exact-terminology queries, but this directional pattern cannot be established statistically with only 5 queries.
2. **Runtime:** BioBERT's one-time model-load cost was 23.0s (first query, Q1); steady-state warm-query latency over the remaining 4 queries averaged 0.203s, approximately 7.8× slower than TF-IDF's 0.026s per query. The blended per-query mean of 4.8s reported below includes the one-time load cost and should not be read as steady-state latency.
3. **Recommendation:** Hybrid two-stage pipeline optimal for production — TF-IDF for fast candidate retrieval, BioBERT for re-ranking on queries where exact lexical overlap is weak.

## Per-Query Results

| Query | Method | GCS | Score | Runtime(s) | Error |
|-------|--------|-----|-------|------------|-------|
| Q1 | tfidf | 156 | 0.0705 | 0.027 | — |
| Q1 | biobert | 196 | 0.6688 | 22.988 | — |
| Q2 | tfidf | 328 | 0.0398 | 0.029 | — |
| Q2 | biobert | 257 | 0.6268 | 0.202 | — |
| Q3 | tfidf | 78 | 0.0427 | 0.019 | — |
| Q3 | biobert | 189 | 0.6161 | 0.191 | — |
| Q4 | tfidf | 273 | 0.0760 | 0.026 | — |
| Q4 | biobert | 293 | 0.6929 | 0.206 | — |
| Q5 | tfidf | 60 | 0.0437 | 0.028 | — |
| Q5 | biobert | 143 | 0.6008 | 0.214 | — |

## Methodology

- **Corpus:** PubMed papers fetched per query (max 20)
- **Primary metric:** Graph Connectivity Score (GCS) = unique gene-disease edges in top-10 result subgraph
- **Secondary metrics:** Score spread, hub gene/disease count, wall-clock runtime
- **Independence:** GCS computed from external knowledge graph, not ranker scores

## Statistical Check

- **Paired queries (n=5):** BioBERT wins=4, TF-IDF wins=1, ties=0
- **Mean GCS delta (BioBERT - TF-IDF):** 36.60 (95% bootstrap CI: -22.00 to 85.60)
- **Exact sign-test p-value (ties excluded):** 0.3750
- **Interpretation:** No statistically significant difference detected at alpha=0.05 — the 95% CI spans zero. Do not interpret the higher point estimate as a demonstrated winner at this sample size.

## Runtime Detail (Cold-Start vs Warm-Query)

- **BioBERT cold start (model load, query Q1):** 23.0s (one-time per process)
- **BioBERT warm-query mean (4 queries: Q2, Q3, Q4, Q5):** 0.203s
- **TF-IDF mean runtime:** 0.026s
- The commonly-quoted blended ratio (BioBERT mean / TF-IDF mean, 183× in the raw JSON summary) includes the one-time model-load cost and overstates steady-state per-query latency. The warm-query ratio above is the correct figure for steady-state comparisons.

## Limitations

- No human relevance judgements (GCS is an automated proxy)
- CPU-only BioBERT - runtime would differ significantly on GPU
- 5 queries is a small evaluation set; results may not generalise

---
*Generated automatically by BioSearch AI evaluator.py*