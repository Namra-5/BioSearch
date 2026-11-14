---
license: cc-by-4.0
language:
- en
pretty_name: BioSearch Retrieval Benchmark
size_categories:
- n<1K
task_categories:
- text-retrieval
tags:
- biomedical
- retrieval-benchmark
- reranking
- pubmed
- trec
- beir
- qrels
configs:
- config_name: corpus
  data_files: beir/corpus.jsonl
- config_name: queries
  data_files: beir/queries.jsonl
- config_name: per_query_results
  data_files: eval_results.csv
---

# BioSearch Retrieval Benchmark

A 25-query biomedical literature **re-ranking** benchmark with hand-labeled binary relevance judgments, used to compare lexical (TF-IDF) and
semantic (BioBERT) ranking in the open-source BioSearch AI pipeline (https://github.com/Namra-5/BioSearch, software DOI 10.5281/zenodo.22078606).

## Scope
For each query, up to 20 papers per source were fetched from PubMed (PubMed only; bioRxiv was not used), both rankers ordered that candidate set,
and the union of their top-10 lists was judged. It is not full-corpus retrieval, and PubMed results drift over time, so the candidate sets are frozen only through these files.

## Dataset structure
```
beir/corpus.jsonl        {"_id": "pubmed:<PMID>", "title": ..., "text": <abstract or "">, "metadata": {source, paper_id, url}}
beir/queries.jsonl       {"_id": "Q1", "text": "BRCA1 BRCA2 breast cancer hereditary mutations"}
beir/qrels/test.tsv      header "query-id<TAB>corpus-id<TAB>score"; score is 0 (not relevant) or 1 (relevant)
qrels.tsv                the same judgments in TREC format (no header): qid  0  docid  grade
runs/tfidf.run, runs/biobert.run   top-10 rankings, TREC run format; score = 1/rank (only the order is meaningful)
queries.json(l)          25 queries with category and origin (Q1-Q5 = original standard queries, Q6-Q25 = added)
candidates.csv           qid, docid, source, paper_id, title, url of every judged paper
eval_results.csv         per-query P@10, MRR@10, nDCG@10 for each system
eval_summary.csv         means, bootstrap 95% CIs, paired Wilcoxon signed-rank and exact sign-test p-values and the TF-IDF minus BioBERT difference
label_provenance.json    number of labels by source
label_review/           calibration, audit and review records
scripts/                 export_benchmark.py, label_workflow.py, run_benchmark_queries.py, make_manifest.py, merge_judge_answers.py
evaluate.py              the evaluator (benchmark/evaluate.py in the GitHub repo)
MANIFEST.json            SHA-256 of every file
```
Document ids are `<source>:<paper_id>` (PMID for PubMed, DOI for bioRxiv). Query ids keep the repository's `Q1`..`Q25` form.

## Loading
```python
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

repo = "Namra-5/biosearch-retrieval-benchmark"
corpus = load_dataset(repo, "corpus", split="train")
queries = load_dataset(repo, "queries", split="train")
qrels_path = hf_hub_download(repo, "beir/qrels/test.tsv", repo_type="dataset")
qrels = pd.read_csv(qrels_path, sep="\t")            # columns: query-id, corpus-id, score
```
Evaluate your own ranking (TREC run file):
`python benchmark/evaluate.py --qrels qrels.tsv --run system=system.run --run tfidf=runs/tfidf.run --out-dir results`

## How the relevance labels were produced
All 25 queries are hand-labeled. Labels are binary and follow a strict user-intent policy: a paper is relevant only if it substantively
satisfies every named component of the query (for example, a TP53-paralog paper is not relevant to a TP53 query). Q1-Q5 were labeled by hand
directly. For Q6-Q25, each paper was judged individually against the same policy, and the judgments were recorded by adopting labels proposed
by a language model (agreement with the hand labels on Q1-Q5: kappa 0.74, accuracy 0.91) after confirming that they
matched the hand reading. A blind random sample of 27 papers was additionally hand-labeled without sight of the proposed labels; all 27 agreed.
The 11 papers with no abstract were labeled not relevant by rule. Counts per label source are in `label_provenance.json`.

## Metrics
Precision@10 (denominator fixed at 10), MRR@10, nDCG@10 (binary gain, log2 discount); judgments exist only for the pooled top-10, so everything is computed within the top 10.
Systems are compared per query with a paired two-sided Wilcoxon signed-rank test (zero differences dropped; normal approximation when scores tie) plus an exact two-sided sign test and a 1,000-resample percentile bootstrap over queries (fixed seed) for 95% CIs.

## Limitations
* 25 queries: confidence intervals are wide; small differences between rankers will not be detectable.
* Re-ranking over a small candidate pool; a system that retrieves other papers has unjudged results counted as non-relevant.
* Single annotator; no inter-annotator agreement is reported. Q1-Q5 were judged from 400-character abstract previews, Q6-Q25 from 1,500-character previews, and 11 papers without an abstract were labeled not relevant by rule.
* Queries were written by the dataset author (gene, disease, drug and mechanism-centred) across about 20 biomedical areas, not sampled from user logs; Q1-Q4 are cancer queries inherited from the original evaluation set. Not for clinical decision-making.
* Titles and identifiers only: abstracts were omitted from this release (`text` in `corpus.jsonl` is empty); retrieve them from PubMed by PMID. Abstract text remains subject to the terms of its publisher/source.

## License
Annotations, queries, run files and scripts: MIT. Third-party titles/abstracts keep their original terms.” with: “Annotations, queries and run files: CC BY 4.0. Evaluation script (evaluate.py): MIT. Third-party titles keep their original terms.

## Citation
```bibtex
@misc{basharat2026biosearchbenchmark,
  author    = {Basharat, Namra},
  title     = {BioSearch Retrieval Benchmark: a 25-query biomedical re-ranking benchmark},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24218563},
  url       = {https://doi.org/10.5281/zenodo.24218563}
}
```
