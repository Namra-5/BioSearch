# Benchmark workflow (maintainer notes)

(The Hugging Face dataset card is `README.md`; this file documents how the benchmark is built.)

Built on top of the existing hand-labeled pipeline: `data/relevance_Q*.csv` -> `scripts/export_benchmark.py` -> standard
TREC `qrels.tsv` + runs -> `benchmark/evaluate.py` -> `eval_results.csv`, `eval_summary.csv`.

| Step | Command | Notes |
|---|---|---|
| 1. Fetch, rank, pool Q6-Q25 | `python scripts/run_benchmark_queries.py` | needs `.env` NCBI_EMAIL, network, BioBERT weights |
| 2a. Calibration gate | `python scripts/label_workflow.py prepare --calibration`, then `compare --file ...` (kappa >= 0.6 vs your Q1-Q5 labels) | |
| 2b. Import labels | `python scripts/merge_judge_answers.py r1.txt r2.txt --out judge_answer.csv`, then `label_workflow.py import --file judge_answer.csv --dry-run`, then without `--dry-run` | validates every id; never overwrites a label |
| 2c. Audit, flagged rows, finalize | `audit make` (label the blind sample), `audit score`, `review make --flagged-only` (fill `human_label`), `review apply`, `accept --no-abstract-as-zero` | full alternative: `review make` / `review apply` for every row |
| 3. Export (TREC + BEIR) | `python scripts/export_benchmark.py --abstracts none` | refuses unreviewed labels |
| 4. Evaluate | `python benchmark/evaluate.py --qrels benchmark/qrels.tsv --run tfidf=benchmark/runs/tfidf.run --run biobert=benchmark/runs/biobert.run --out-dir benchmark` | |
| 5. Checksums | `python scripts/make_manifest.py` | rerun after any change |

Reproduction check (runs in CI): starting from the committed labels, the pipeline reproduces the per-query P@10 stored in
`data/precision_recall_summary.json` (TF-IDF mean 0.84, BioBERT mean 0.80 on Q1-Q5).

## Scope and honest limitations
* Re-ranking benchmark: each query's candidate set is the <=20 papers per source that PubMed returned at fetch time; the rankers
  reorder that small set. It is not full-corpus retrieval, and the candidate set is frozen only through the committed CSVs / `candidates.csv`.
* Binary relevance (0/1) under a strict user-intent policy; nDCG@10 therefore uses binary gain.
* Single annotator; judgments exist only for the pooled top-10 of the two systems, so MRR@10 and nDCG@10 are computed within the top 10.
* With 25 queries confidence intervals are wide; expect small differences to be undetectable.
