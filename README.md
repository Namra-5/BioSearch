# BioSearch AI

Week 1 of BioSearch AI is complete.

## What it does

- Fetches biomedical papers from PubMed and bioRxiv.
- Caches results in SQLite to avoid repeated API calls.
- Ranks papers with a TF-IDF baseline.
- Supports CLI options for `--stats`, `--json`, `--quiet`, and `--prune-days`.

## Current status

- `src/models.py`: validated data models.
- `src/fetcher_pubmed.py`: PubMed Entrez client with retries and rate limiting.
- `src/fetcher_biorxiv.py`: bioRxiv REST client with adaptive local filtering.
- `src/storage.py`: SQLite cache with query normalization and pruning.
- `src/ranker_tfidf.py`: baseline ranking and safe fallback behavior.
- `main_week1.py`: command-line entry point.

## Requirements

Install the Week 1 dependencies:

```bash
pip install -r requirements.txt
```

## NCBI setup

PubMed access requires a valid email address. An NCBI API key is optional but recommended because it increases request limits. Create or sign in to an NCBI account, generate an API key, and add it to `.env` as:

```env
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=your_api_key
```

## Example commands

```bash
python main_week1.py --query "BRCA1 breast cancer" --source pubmed
python main_week1.py --query "CRISPR genomics" --source biorxiv --max 5
python main_week1.py --stats
```

## Notes

- Keep the `tests/` folder. It is useful for regression checks and future weeks.
- This repository is intended to grow into semantic search, NER, and graph extraction in later weeks.