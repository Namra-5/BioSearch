# BioSearch AI— Human-Validated Precision/Recall Findings

**This file is hand-reviewed and NOT auto-regenerated.** It is separate from
`findings.md` (which reports the automated GCS/statistical evaluation and
is safely overwritten by `python main.py --evaluate`). This file must be
updated manually if the underlying relevance CSVs or evaluator output
change — do not wire it into any auto-generation script.

## 1. Evaluation Status

- Evaluation completed on 5 of 5 standard queries (Q1–Q5).
- Both TF-IDF and BioBERT were evaluated over the same query set.
- Cutoff: top-10 results per method.
- Relevance labels were manually assigned by the project author using a
  strict user-intent-satisfaction policy (Section 5).
- This is a small, manually-labeled evaluation set intended to
  cross-validate the automated GCS metric reported in `findings.md` — it
  is **not** a comprehensive retrieval benchmark.

## 2. Evaluation Methodology

- **P@10** — relevant documents in a method's top-10, divided by a fixed
  denominator of 10.
- **Pooled Recall@10** — relevant documents found in a method's top-10,
  divided by the total number of relevant documents found anywhere in the
  **union** of TF-IDF's and BioBERT's top-10 result sets for that query
  (standard TREC-style pooling).
- Unranked/extra pooled rows in the source CSVs are excluded from both
  the numerator and denominator.
- Incomplete relevance labels are rejected by the evaluator — every row
  used in a computation has a validated `0`/`1` label.
- Each method's top-10 was validated to contain unique ranks `1..10`
  with no duplicate paper identities.

## 3. Important Limitation

**The pooled Recall@10 metric is a pool-based estimate.** The denominator
consists of relevant papers identified within the union of the two
systems' top-10 results; it is **not** an exhaustive count of all relevant
papers in the biomedical literature for that query. A paper relevant to
the query but absent from both systems' top-10 is invisible to this
metric by construction.

Only five queries were labeled and evaluated. No statistical significance
test was run on precision/recall (unlike the GCS metric in `findings.md`,
which does carry a paired sign test / bootstrap CI) — with n=5, such a
test would carry the same statistical-power caveats already documented
there, and is not claimed here.

## 4. Relevance Policy — Strict User-Intent Satisfaction

A paper was judged relevant only if it substantively satisfied the
query's actual information need, not merely because it shared a keyword
or a broad topic with the query. Representative cases:

- **Q3 (TP53):** a paper primarily about p73 — a TP53 paralog, not TP53
  itself — was judged not relevant despite topical proximity.
- **Q5 (APOE):** papers covering Alzheimer's disease and amyloid broadly
  but omitting APOE were judged not relevant, since APOE was an explicit,
  named component of the query.
- **Q2 (KRAS + lung cancer + targeted therapy):** a paper about KRAS
  without a lung-cancer focus, and a paper about EGFR/EREG rather than
  KRAS, were both judged not relevant — all three named constraints had
  to be satisfied together.
- **Q4 (EGFR-TKI resistance + NSCLC):** a generic tyrosine-kinase-inhibitor
  resistance paper that did not specifically establish EGFR was judged
  not relevant.

This is a deliberate, disclosed methodological choice, not a labeling
error — it trades recall for precision in the judgments themselves, which
is why the resulting Recall@10 figures should be read as "recall against
a strict definition of relevance," not a lenient one.

## 5. Per-Query Results

| Query | TF-IDF P@10 | TF-IDF Pooled Recall@10 | BioBERT P@10 | BioBERT Pooled Recall@10 | Pooled Relevant |
|-------|------------:|------------------------:|-------------:|--------------------------:|-----------------:|
| Q1    | 0.9         | 0.8182                  | 0.9          | 0.8182                    | 11                |
| Q2    | 0.8         | 0.8000                  | 0.7          | 0.7000                    | 10                |
| Q3    | 0.8         | 0.7273                  | 1.0          | 0.9091                    | 11                |
| Q4    | 0.9         | 0.6923                  | 1.0          | 0.7692                    | 13                |
| Q5    | 0.8         | 1.0000                  | 0.4          | 0.5000                    | 8                 |

## 6. Aggregate Results

| Method  | Mean P@10 | Mean Pooled Recall@10 |
|---------|----------:|------------------------:|
| TF-IDF  | 0.84      | 0.8076                  |
| BioBERT | 0.80      | 0.7393                  |

Across the five labeled queries, TF-IDF had higher mean P@10 and pooled
Recall@10 than BioBERT. This is a within-this-evaluation-set finding, not
a claim of general superiority — see Section 8.

## 7. Query-Level Observations

- **Q1:** TF-IDF and BioBERT performed identically (P@10=0.9, pooled
  recall=0.8182 for both).
- **Q2:** TF-IDF outperformed BioBERT (0.8 vs 0.7 P@10).
- **Q3:** BioBERT outperformed TF-IDF (1.0 vs 0.8 P@10).
- **Q4:** BioBERT outperformed TF-IDF (1.0 vs 0.9 P@10).
- **Q5:** TF-IDF substantially outperformed BioBERT (0.8 vs 0.4 P@10).

No causal explanation for these per-query differences is asserted here
beyond what is directly supported by the retrieved titles — see
`data/relevance_Q*.csv` for the underlying labeled candidates.

## 8. Interpretation

- TF-IDF performed better in aggregate on this five-query, manually
  labeled evaluation set.
- BioBERT outperformed TF-IDF specifically on Q3 and Q4.
- Results are query-dependent: neither method dominated on every query.
- Five queries are an insufficient sample for a general claim that either
  retrieval method is superior; this is a finding about this evaluation
  set, not a proof of universal superiority.
- This precision/recall finding is **directionally consistent** with, but
  methodologically independent from, the GCS-based evaluation in
  `findings.md`, which found no statistically significant aggregate
  difference between the two methods (sign-test p=1.00, n=5, 95% CI
  spanning zero) — GCS measures retrieved-set biomedical graph richness,
  while this file measures strict-intent relevance, and they are not
  expected to agree in every particular.

## 9. Limitations

- Only five labeled queries.
- Manual relevance judgments made by a single annotator (the project
  author) — no inter-annotator agreement was measured.
- Pool-based, not exhaustive, recall.
- Judgment subjectivity is possible despite the documented strict-intent
  policy; a different annotator applying the same policy might disagree
  on borderline cases.
- Limited query diversity (all five queries are gene/disease-oncology or
  neurodegeneration focused; no rare-disease, drug-repurposing, or
  non-oncology query is represented).
- Top-10 cutoff only — behavior beyond rank 10 is not evaluated.
- No statistical significance test is claimed for the precision/recall
  numbers in this file.

## 10. Reproducibility

Input files:
```
data/relevance_Q1.csv
data/relevance_Q2.csv
data/relevance_Q3.csv
data/relevance_Q4.csv
data/relevance_Q5.csv
```

Evaluator:
```
scripts/compute_precision_recall.py
```

Command used:
```bash
python scripts/compute_precision_recall.py --in "data/relevance_Q*.csv"
```

Output summary:
```
data/precision_recall_summary.json
```

## 11. Validation Performed

- All five relevance CSVs passed the evaluator's built-in validation:
  complete, non-empty `0`/`1` relevance labels on every scored row.
- Each method's top-10 was confirmed to contain unique ranks `1..10`
  with no duplicate paper identities.
- Pooled relevant counts were independently re-verified by hand against
  each query's reported P@10 and pooled Recall@10 (arithmetic
  cross-check: `relevant_in_top10 = P@10 × 10`, and
  `recall = relevant_in_top10 / pooled_relevant`) — all five queries were
  internally consistent to 4 decimal places.
- Benchmark CSVs and the evaluator script were not modified as part of
  producing this document.

---
*This document reports a manually-labeled cross-validation of the
automated GCS evaluation in `findings.md`. It is maintained by hand and
is not regenerated by any script — update it manually if the underlying
CSVs change.*
