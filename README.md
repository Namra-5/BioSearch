<div align="center">

# BioSearch AI

**A biomedical literature retrieval system that rigorously compares lexical (TF-IDF) and semantic (BioBERT) search, backed by a gene-disease knowledge graph and a statistically honest evaluation.**

[![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/github/actions/workflow/status/Namra-5/BioSearch/tests.yml?branch=main&label=tests)](https://github.com/Namra-5/BioSearch/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

Choosing between fast keyword search and slower semantic search for biomedical literature is usually assumed rather than measured. BioSearch AI measures it directly: it retrieves literature from PubMed and bioRxiv, ranks it with both classical lexical search and BioBERT-based semantic search, extracts genes and diseases with a hybrid NER pipeline, builds a co-occurrence knowledge graph, and then evaluates which approach actually performs better — with paired statistical testing and a separate hand-labeled precision/recall benchmark, and it reports the result honestly even when the answer is "no measurable difference."

It started as a Week 1 lexical-retrieval exercise and grew, week by week, into a small research project: by Week 4 the question stopped being "does it run?" and became "what can I actually prove about it?"

---

## Table of Contents

- [Key Features](#key-features)
- [Project Evolution](#project-evolution-weeks-1-4)
- [Evaluation & Findings](#evaluation--findings)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Quick Usage Examples](#quick-usage-examples)
- [CLI Reference](#cli-reference)
- [Web Dashboard](#web-dashboard)
- [Data & Reproducibility](#data--reproducibility)
- [Repository Structure](#repository-structure)
- [Testing](#testing)
- [Limitations & Honest Scope](#limitations--honest-scope)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Key Features

- **Dual-source retrieval** — PubMed (peer-reviewed) and bioRxiv (preprint) literature, queried independently or together. When both sources are used, results are automatically deduplicated so the same underlying work isn't double-counted just because it exists as both a preprint and a later peer-reviewed record.
- **Dual-mode ranking** — TF-IDF (fast, exact-terminology-strong) and BioBERT (semantic, synonym/concept-aware), run side by side or independently.
- **Hybrid biomedical NER** — rule-based `EntityRuler` patterns combined with a statistical spaCy model to extract genes and diseases from free text, augmented with NLM's own curated MeSH terms for higher-precision disease detection.
- **Gene-disease knowledge graph** — built with NetworkX from entity co-occurrence across retrieved papers, with centrality, hub-node, and density analytics.
- **Statistically honest evaluation** — a custom Graph Connectivity Score (GCS) compared across methods with a paired sign test and bootstrap confidence interval, plus a separate, hand-labeled Precision@10 / pooled Recall@10 benchmark for cross-validation — with explicit, disclosed limitations rather than overclaimed wins.
- **Local SQLite caching** (WAL mode) for papers and embeddings, so repeated queries don't re-hit PubMed's rate-limited API.

---

## Project Evolution (Weeks 1-4)

| Week | Focus | What was added |
|------|-------|-----------------|
| **1 — Lexical foundations** | Ingestion & baseline retrieval | PubMed/bioRxiv fetchers, SQLite caching, TF-IDF ranking, Pydantic data models |
| **2 — Semantic retrieval** | Meaning over exact terms | BioBERT (`sentence-transformers`) embeddings, persistent embedding cache, TF-IDF vs BioBERT comparison |
| **3 — Knowledge extraction** | From text to structure | Hybrid NER (rules + statistical spaCy), gene-disease knowledge graph, centrality/subgraph analytics, JSON/edgelist export |
| **4 — Evaluation & hardening** | From "it runs" to "it's proven" | Unified CLI (`main.py`), Graph Connectivity Score evaluator, paired sign test + bootstrap CI, MeSH-term fusion, cross-source dedup, WAL-mode SQLite, CI pipeline, hand-labeled Precision/Recall benchmark, web dashboard |

Each phase is additive — Week 4's evaluation and hardening work sits on top of the Week 1-3 pipeline without replacing it.

---

## Evaluation & Findings

BioSearch AI is evaluated two independent ways, deliberately kept in two separate documents with two different reproducibility rules, because they behave differently.

### Graph Connectivity Score — live, auto-generated (`findings.md`)

Five standardized biomedical queries, each run through both TF-IDF and BioBERT, scored by how many unique gene-disease edges their top-10 results produce in the knowledge graph, with a paired sign test and bootstrap 95% confidence interval.

Using this paired statistical test across the five standard queries, the evaluation has consistently found **no statistically significant difference** between TF-IDF and BioBERT on GCS at this sample size — the confidence interval for the mean delta spans zero. BioBERT's one-time model-load cost is reported separately from its steady-state warm-query latency, to avoid the misleading blended average a naive mean would otherwise produce.

This file is regenerated every time `python main.py --evaluate` runs (or via the Web Dashboard's Live Evaluation tab), so it reflects the current state of the corpus and knowledge graph — including changes from MeSH-term fusion and cross-source deduplication, both of which intentionally alter which entities are available to the graph and will shift the exact numbers from any earlier snapshot. **Treat `findings.md` itself, not any number quoted here or elsewhere, as the current source of truth.**

Full report: [`findings.md`](findings.md) · Raw data: [`data/comparison_results.csv`](data/comparison_results.csv)

### Precision & Recall — human-labeled, frozen (`PRECISION_RECALL_FINDINGS.md`)

Five queries, hand-labeled by the project author under a strict user-intent-satisfaction relevance policy (topical overlap alone does not count as relevant — see the file for worked examples). Unlike GCS, these numbers do not drift between runs: they are fixed unless the underlying hand-labeled CSVs are deliberately revised.

| Method | Mean P@10 | Mean pooled Recall@10 |
|--------|----------:|------------------------:|
| TF-IDF | 0.84 | 0.8076 |
| BioBERT | 0.80 | 0.7393 |

TF-IDF scored higher in aggregate on this five-query set; BioBERT specifically outperformed it on 2 of 5 queries. The two evaluations are methodologically independent and are not expected to agree in every particular — GCS measures retrieved-set graph richness, while this benchmark measures strict relevance judged by a human.

Full report: [`PRECISION_RECALL_FINDINGS.md`](PRECISION_RECALL_FINDINGS.md)

> Both documents are explicit about their limitations (small *n*, single annotator, pooled rather than exhaustive recall). Neither claims one retrieval method is universally superior — see [Limitations & Honest Scope](#limitations--honest-scope).

---

## Architecture & Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Retrieval sources** | PubMed E-utilities, bioRxiv REST API | Peer-reviewed and preprint biomedical literature |
| **Storage** | SQLite (WAL mode) | Query/paper/embedding caching |
| **Lexical ranking** | scikit-learn `TfidfVectorizer` | Fast, exact-terminology retrieval |
| **Semantic ranking** | `sentence-transformers` (BioBERT, 768-d) | Concept/synonym-aware retrieval |
| **NER** | spaCy (`EntityRuler` + statistical model) | Gene/disease entity extraction |
| **Domain vocabulary** | NLM MeSH headings | High-precision disease-entity augmentation |
| **Knowledge graph** | NetworkX | Gene-disease co-occurrence graph & analytics |
| **Data models** | Pydantic | Validated `Paper`, `SearchResult`, etc. |
| **Evaluation** | Custom evaluator with a paired sign test and bootstrap CI | Statistical comparison of retrieval methods |
| **CI** | GitHub Actions | Automated test gate on push/PR, installed from the pinned lockfile |
| **Web dashboard** | Streamlit | Live search, live evaluation, and read-only findings display |

---

## Getting Started

### Prerequisites

- Python 3.13 (developed and tested on 3.13.14; CI runs on 3.13)
- A free [NCBI account](https://www.ncbi.nlm.nih.gov/account/) email for PubMed API access (an API key is optional but recommended — raises your rate limit)
- ~1.5 GB free disk space (BioBERT model weights + spaCy model)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Namra-5/BioSearch.git
cd BioSearch

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies from the pinned lockfile (this is also what CI installs)
pip install -r requirements-lock.txt

# 4. Install the spaCy statistical model
python -m spacy download en_core_web_sm
```

### Configuration

Create a `.env` file in the project root:

```env
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=your_optional_api_key
```

| Variable | Required | Purpose |
|----------|:--------:|---------|
| `NCBI_EMAIL` | Yes, for PubMed | Required by NCBI's E-utilities usage policy for all PubMed requests |
| `NCBI_API_KEY` | Optional | Raises your PubMed rate limit from roughly 3 to 10 requests/second |

bioRxiv-only searches do not require either variable. `.env` is excluded via `.gitignore` — never commit real credentials.

### Quick Usage Examples

```bash
# Search PubMed with TF-IDF only
python main.py --query "BRCA1 breast cancer" --source pubmed --max 15

# Search bioRxiv preprints only
python main.py --query "BRCA1 breast cancer" --source biorxiv --max 15

# Search both sources at once (automatically deduplicated)
python main.py --query "KRAS lung cancer" --source both --max 10

# Compare both ranking methods
python main.py --query "CRISPR genomics" --method both --max 10

# Extract genes/diseases from top results
python main.py --query "BRCA1 breast cancer" --ner

# Build and inspect a gene-disease knowledge graph
python main.py --query "KRAS lung cancer" --graph --entity kras

# Run the full TF-IDF vs BioBERT evaluation (regenerates findings.md)
python main.py --evaluate
```

---

## CLI Reference

`main.py` is the unified entry point for every capability above.

| Flag | Description |
|------|-------------|
| `-q, --query` | Search query text |
| `--method {tfidf,biobert,both,none}` | Ranking method to use (default: `tfidf`) |
| `--source {pubmed,biorxiv,both}` | Literature source(s) — `both` triggers automatic dedup |
| `-m, --max` | Max papers to fetch per source (default: 15) |
| `--ner` | Run NER extraction on the top results |
| `--rules-only` | Restrict NER to high-precision rule patterns only |
| `--graph` | Build a knowledge graph from results |
| `--entity <name>` | Show 1-hop neighbours of an entity in the graph |
| `--save-graph <path>` | Export the graph as JSON |
| `--save-edgelist <path>` | Export the graph as a TSV edgelist (Gephi/Cytoscape) |
| `--evaluate` | Run the full TF-IDF vs BioBERT evaluation and regenerate `findings.md` |
| `--findings <path>` | Output path for the evaluation report (default: `findings.md`) |
| `--no-biobert` | Skip BioBERT during evaluation (TF-IDF only) |
| `--compare A B` | Compute BioBERT semantic similarity between two text snippets |
| `--stats` | Show cache/storage statistics |
| `--embed-stats` | Show embedding cache statistics |
| `--prune-days N` | Remove cached entries older than N days |
| `--no-cache` | Bypass the local cache for this run |
| `--json` | Emit machine-readable JSON output |
| `--quiet` | Suppress non-essential console output |
| `--device {cpu,cuda}` | Force BioBERT device selection |

---

## Web Dashboard

A Streamlit dashboard (`app.py`) sits over the same underlying classes as `main.py` — no separate implementation, no logic duplication.

```bash
# Streamlit is included in requirements-lock.txt
streamlit run app.py
```

Four tabs:

- **Search** — live retrieval and ranking against PubMed, bioRxiv, or both, with the same source and dedup behavior as the CLI.
- **Live Evaluation (GCS)** — runs the real evaluator on demand and regenerates `findings.md` on disk, so the dashboard always reflects a genuine run rather than a cached snapshot.
- **Precision and Recall** — a read-only view of the hand-labeled benchmark, visually and textually marked as ground truth this app cannot regenerate.
- **About** — a short explanation of exactly which underlying function each tab calls.

---

## Data & Reproducibility

`data/` is gitignored by default, with six explicit evidence-file exceptions intended to be tracked in version control:

```gitignore
data/*
!data/relevance_Q1.csv
!data/relevance_Q2.csv
!data/relevance_Q3.csv
!data/relevance_Q4.csv
!data/relevance_Q5.csv
!data/precision_recall_summary.json
```

The five `relevance_Q*.csv` files and `precision_recall_summary.json` are the hand-labeled ground truth behind `PRECISION_RECALL_FINDINGS.md` — they are committed so a fresh clone can reproduce that file's numbers exactly via `python scripts/compute_precision_recall.py --in "data/relevance_Q*.csv"`. Everything else under `data/` (`comparison_results.csv`, `findings_summary.json`, the paper/embedding cache) is safe to regenerate live and is intentionally not committed.

---

## Repository Structure

```
biosearch/
├── main.py                          # Unified CLI entry point
├── config.py                        # Environment/config loading
├── app.py                           # Streamlit web dashboard
├── findings.md                      # Auto-generated GCS evaluation report
├── PRECISION_RECALL_FINDINGS.md     # Hand-maintained precision/recall report
├── requirements.txt
├── requirements-lock.txt            # Pinned, reproducible dependency versions (used by CI)
├── .github/workflows/tests.yml      # CI test gate
├── src/
│   ├── models.py                    # Pydantic data models
│   ├── fetcher_pubmed.py            # PubMed E-utilities client
│   ├── fetcher_biorxiv.py           # bioRxiv REST client
│   ├── storage.py                   # SQLite caching (WAL mode)
│   ├── ranker_tfidf.py              # TF-IDF ranking
│   ├── ranker_semantic.py           # BioBERT ranking + embedding cache
│   ├── embedder.py                  # BioBERT embedding generation
│   ├── ner_extractor.py             # Hybrid NER (rules + statistical)
│   ├── mesh_fusion.py               # MeSH-term disease augmentation
│   ├── dedup.py                     # Cross-source paper deduplication
│   ├── knowledge_base.py            # Entity to graph pipeline
│   ├── knowledge_graph.py           # NetworkX graph construction & analytics
│   └── evaluator.py                 # GCS evaluation + statistical testing
├── scripts/
│   ├── build_relevance_annotation_template.py
│   ├── compute_precision_recall.py
│   └── verify_phase1_hardening.py
├── tests/                           # pytest suite (190+ tests)
└── data/                            # Cached papers + evaluation data (see Data & Reproducibility)
```

---

## Testing

```bash
# Full suite
pytest -q

# Fast subset (no BioBERT/spaCy load required)
pytest -m "not slow and not integration" -q
```

The suite covers fetchers, storage, both rankers, NER (including gene-symbol canonicalization edge cases), the knowledge graph, deduplication, MeSH fusion, and the evaluator's statistical functions — all against fake/injected dependencies, so the full suite runs in well under two minutes with no network access required. CI installs from `requirements-lock.txt` on Python 3.13 and runs the fast subset on every push.

---

## Limitations & Honest Scope

This project is intended to demonstrate rigorous engineering and evaluation judgment under real time constraints — not to be a production-scale search engine. Documented, deliberate scope decisions:

- **No full UMLS integration** — MeSH-term fusion uses a lightweight, curated heuristic instead of a licensed ontology API; a documented precision/recall trade-off, not an oversight.
- **GCS values are not fixed** — adding MeSH-term fusion and cross-source deduplication intentionally changed which entities the knowledge graph sees, so GCS numbers from an earlier run are not comparable to a later one without re-running the evaluation. This is expected behavior of a metric computed from a live, evolving graph, not an inconsistency in the metric itself — see [Evaluation & Findings](#evaluation--findings).
- **Small evaluation sample (n=5 queries)** — both evaluation methods explicitly report this and avoid claiming statistical significance where none exists.
- **Pooled, not exhaustive, recall** — Recall@10 is measured against the union of both methods' top-10 results, per standard TREC-style pooling methodology, not the full biomedical corpus.
- **Single annotator** — precision/recall relevance labels were judged by the project author under a disclosed, strict policy; no inter-annotator agreement was measured.
- **CPU-only BioBERT benchmarking** — GPU latency would differ substantially from the reported numbers.
- **No production infrastructure** (async fetching, a real vector index, multi-user support) — these are named as future work rather than built prematurely; see [Roadmap](#roadmap).

---

## Roadmap

- Full UMLS/semantic-type integration for entity normalization
- Expand the evaluation query set beyond n=5 for stronger statistical power
- FAISS/pgvector-backed similarity search at larger corpus scale
- PostgreSQL migration path for multi-user/concurrent access
- Persistent inverted index (FTS5) instead of per-query TF-IDF refitting
- Expand hand-labeled precision/recall coverage beyond the current 5 queries
- Evaluate retrieval quality under low-bandwidth or rate-limited API conditions, relevant to institutions without a paid NCBI key

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Open an issue describing the bug or feature before submitting a large PR.
2. Fork the repo and create a feature branch: `git checkout -b feature/your-feature`.
3. Make sure `pytest -q` passes locally before opening a PR.
4. Keep PRs focused — one logical change per PR is easier to review and merge.

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Contact

**Namra Basharat**
[github.com/Namra-5/BioSearch](https://github.com/Namra-5/BioSearch)

For questions about the methodology, design decisions, or potential collaboration, feel free to open an issue or reach out directly.
