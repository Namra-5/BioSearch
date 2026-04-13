# BioSearch AI

BioSearch AI is a modular biomedical search system evolving from lexical retrieval to semantic embedding-based intelligence.

---

## System Overview

- **Week 1 (Lexical):** Focused on robust data ingestion from PubMed/bioRxiv, SQLite-based query caching, and TF-IDF baselines. 
- **Week 2 (Semantic):** Advanced retrieval using BioBERT (768-d vectors) with persistent vector caching and comparative ranking logic.

---

## What it does

- Fetches biomedical papers from PubMed and bioRxiv.
- Caches query results in SQLite to avoid redundant API calls.
- Stores embeddings as 768-dimensional BioBERT vectors.
- Supports both lexical (TF-IDF) and semantic (BioBERT) ranking.
- Enables side-by-side comparison of ranking strategies.

---

## Current Status

- `src/models.py`: validated data models. 
- `src/fetcher_pubmed.py`: PubMed Entrez client with retries and rate limiting. 
- `src/fetcher_biorxiv.py`: bioRxiv REST client with adaptive local filtering.
- `src/storage.py`: SQLite cache with query normalization and pruning.
- `src/ranker_tfidf.py`: baseline ranking and safe fallback behavior. 
- `main_week1.py`: command-line entry point.

---

## Technical Architecture

- **Fetch Layer:** PubMed + bioRxiv API clients.
- **Storage Layer:** SQLite caching for papers and embeddings.
- **Embedding Layer:** BioBERT (768-d dense vector representations).
- **Ranking Layer:** Hybrid approach(Sparse TF-IDF vs Dense Semantic BioBERT).

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Install NLP backend:

```bash
pip install sentence-transformers torch
```

Optional GPU acceleration:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Configuration

Create a `.env` file:

```env
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=your_api_key
```

---

## Example Usage

```bash
# Semantic ranking
python main_week2.py --query "BRCA1 breast cancer" --method biobert

# Compare TF-IDF vs BioBERT
python main_week2.py --query "CRISPR genomics" --method both --max 10

# Semantic similarity between terms
python main_week2.py --compare "myocardial infarction" "heart attack"
```

---

## Testing

All modules are validated using `pytest`:

* Unit tests for embedding cache integrity
* Integration tests for embedding pipeline
* Robustness tests for edge cases (empty inputs, long abstracts)

Run tests:

```bash
pytest
```

---

## Notes

* Week 1 focuses on lexical retrieval and caching.
* Week 2 introduces semantic understanding via BioBERT embeddings.
* Embedding cache eliminates redundant computation across runs.
* System is designed to extend into biomedical knowledge graphs and NER in later phases.

```
