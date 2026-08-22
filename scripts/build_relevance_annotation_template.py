"""
scripts/build_relevance_annotation_template.py 

Build a pooled relevance-annotation CSV for a query.

The script retrieves and ranks papers with TF-IDF and BioBERT, pools the
union of their top-N results, and writes an empty ``relevant`` column for
manual 0/1 judgments. It does not generate or infer relevance labels.

Example::

     python scripts/build_relevance_annotation_template.py \
          --query "BRCA1 BRCA2 breast cancer hereditary mutations" \
          --query-id Q1 \
          --out data/relevance_Q1.csv

The resulting CSV can be processed by ``compute_precision_recall.py``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make `src` importable when run as `python scripts/build_relevance_annotation_template.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a relevance-labeling CSV for one query.")
    p.add_argument("--query", required=True, help="The exact query text to fetch/rank.")
    p.add_argument("--query-id", required=True, help="Short id, e.g. Q1 (for your records).")
    p.add_argument("--max", type=int, default=20, help="Max papers fetched per source.")
    p.add_argument("--top-n", type=int, default=10, help="Top-N per method to pool.")
    p.add_argument("--source", choices=["pubmed", "biorxiv", "both"], default="pubmed")
    p.add_argument("--out", required=True, help="Output CSV path, e.g. data/relevance_Q1.csv")
    p.add_argument("--abstract-chars", type=int, default=400,
                    help="How many abstract characters to preview in the CSV.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    from config import DB_PATH, NCBI_API_KEY, NCBI_EMAIL
    from src.storage import PaperStorage
    from src.fetcher_pubmed import PubMedFetcher
    from src.ranker_tfidf import TFIDFRanker

    if not NCBI_EMAIL:
        print("Error: Set NCBI_EMAIL in .env before running this script.")
        sys.exit(1)

    storage = PaperStorage(db_path=DB_PATH)
    fetchers = []
    if args.source in ("pubmed", "both"):
        fetchers.append(("pubmed", PubMedFetcher(email=NCBI_EMAIL, api_key=NCBI_API_KEY)))
    if args.source in ("biorxiv", "both"):
        from src.fetcher_biorxiv import BioRxivFetcher
        fetchers.append(("biorxiv", BioRxivFetcher()))

    papers = []
    for source_name, fetcher in fetchers:
        cached = storage.get_papers_for_query(args.query, source=source_name)
        if cached:
            papers.extend(cached)
        else:
            fetched = fetcher.fetch(args.query, max_results=args.max)
            storage.cache_query_results(args.query, source_name, fetched)
            papers.extend(fetched)

    if not papers:
        print("No papers fetched — nothing to label. Check your query/network/API key.")
        sys.exit(1)

    # Optional but recommended: dedup before ranking so the same underlying
    # work isn't pooled twice under two source records (see src/dedup.py).
    try:
        from src.dedup import deduplicate_papers
        dedup_result = deduplicate_papers(papers)
        print(dedup_result.summary())
        papers = dedup_result.papers
    except ImportError:
        print("Note: src/dedup.py not found — skipping cross-source dedup for this run.")

    # TF-IDF top-N
    tfidf_ranked = TFIDFRanker().rank(papers, args.query, top_n=args.top_n)
    tfidf_top = [(r.paper, "tfidf", r.rank) for r in tfidf_ranked]

    # BioBERT top-N (heavier — only load if we actually need it)
    try:
        from src.ranker_semantic import SemanticRanker
        bio_ranked = SemanticRanker().rank(papers, args.query, top_n=args.top_n)
        bio_top = [(r.paper, "biobert", r.rank) for r in bio_ranked]
    except Exception as exc:
        print(f"Warning: BioBERT ranking unavailable ({exc}). "
              f"Pooling from TF-IDF only.")
        bio_top = []

    # Pool: union of both methods' top-N, deduplicated by (source, paper_id),
    # but remember EVERY method+rank a pooled paper appeared at, since
    # compute_precision_recall.py needs per-method top-N membership.
    pooled: dict[tuple, dict] = {}
    for paper, method, rank in tfidf_top + bio_top:
        key = (getattr(paper.source, "value", paper.source), paper.paper_id)
        if key not in pooled:
            pooled[key] = {
                "paper": paper,
                "methods": {},
            }
        pooled[key]["methods"][method] = rank

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_id", "query_text", "source", "paper_id", "title",
        "abstract_preview", "url", "tfidf_rank", "biobert_rank",
        "relevant",  # <-- YOU fill this column in with 0 or 1
        "notes",     # <-- optional free text
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (source, paper_id), entry in pooled.items():
            paper = entry["paper"]
            abstract = (paper.abstract or "")[: args.abstract_chars]
            writer.writerow({
                "query_id": args.query_id,
                "query_text": args.query,
                "source": source,
                "paper_id": paper_id,
                "title": paper.title,
                "abstract_preview": abstract,
                "url": paper.url or "",
                "tfidf_rank": entry["methods"].get("tfidf", ""),
                "biobert_rank": entry["methods"].get("biobert", ""),
                "relevant": "",   # left blank intentionally
                "notes": "",
            })

    print(f"\nWrote {len(pooled)} pooled papers to {out_path}")
    print("Next step: open the CSV, read each abstract_preview, and fill the")
    print("'relevant' column with 0 or 1. Do not guess from the title alone.")
    print("Then run: python scripts/compute_precision_recall.py --in "
          f"{out_path}")


if __name__ == "__main__":
    main()
