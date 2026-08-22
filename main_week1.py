# main_week1.py
# Legacy integration CLI for the initial retrieval pipeline.
# Usage: python main_week1.py --query "BRCA1 breast cancer" --max 10

from __future__ import annotations

import argparse
import json
import sys

from pydantic import BaseModel

from config import setup_logging, NCBI_EMAIL, NCBI_API_KEY, DB_PATH
from src.models import SearchQuery
from src.fetcher_pubmed import PubMedFetcher
from src.fetcher_biorxiv import BioRxivFetcher
from src.storage import PaperStorage
from src.ranker_tfidf import TFIDFRanker


def _result_to_dict(result: BaseModel) -> dict:
    paper = result.paper
    return {
        'rank': result.rank,
        'score': result.score,
        'method': result.method,
        'paper': {
            'paper_id': paper.paper_id,
            'source': paper.source,
            'title': paper.title,
            'abstract': paper.abstract,
            'authors': paper.authors,
            'published_date': paper.published_date.isoformat() if paper.published_date else None,
            'doi': paper.doi,
            'journal': paper.journal,
            'keywords': paper.keywords,
            'url': paper.url,
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="BioSearch AI retrieval CLI")
    parser.add_argument('--query', '-q', help='Biomedical search query')
    parser.add_argument('--max', '-m', type=int, default=10, help='Max results per source')
    parser.add_argument('--source', choices=['pubmed', 'biorxiv', 'both'], default='both', help='Data source to search')
    parser.add_argument('--no-cache', action='store_true', help='Bypass cache and fetch data')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON output')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-essential CLI messages')
    parser.add_argument(
        '--prune-days',
        type=int,
        metavar='DAYS',
        help='Prune cache rows older than DAYS and exit',
    )
    args = parser.parse_args()

    setup_logging('WARNING' if args.quiet else 'INFO')

    storage = PaperStorage(db_path=DB_PATH)

    if args.stats:
        print(json.dumps(storage.stats(), indent=2))
        return

    if args.prune_days is not None:
        summary = storage.prune_stale(older_than_days=args.prune_days)
        print(json.dumps(summary, indent=2))
        return

    if not args.query:
        parser.error('the following arguments are required: --query/-q (unless using --stats or --prune-days)')

    if args.source in ('pubmed', 'both') and not NCBI_EMAIL:
        print('Error: Set NCBI_EMAIL in .env file to use PubMed fetcher.')
        sys.exit(1)

    search_query = SearchQuery(query=args.query, max_results=args.max)
    all_papers = []

    # PubMed
    if args.source in ('pubmed', 'both'):
        use_cache = False
        if not args.no_cache and storage.was_recently_fetched(search_query.query, source='pubmed'):
            cached = storage.search_cached(search_query.query, source='pubmed')
            if len(cached) >= search_query.max_results:
                use_cache = True
                all_papers.extend(cached[:search_query.max_results])
                if not args.quiet:
                    print(f'[CACHE] Loaded {search_query.max_results} PubMed papers from cache.')
            else:
                if not args.quiet:
                    print(
                        '[CACHE] PubMed cache has '
                        f'{len(cached)} paper(s), need {search_query.max_results}. Fetching fresh data.'
                    )

        if not use_cache:
            fetcher = PubMedFetcher(email=NCBI_EMAIL, api_key=NCBI_API_KEY)
            papers = fetcher.fetch(query=search_query.query, max_results=search_query.max_results)
            storage.cache_query_results(query=search_query.query, source='pubmed', papers=papers)
            all_papers.extend(papers)
            if not args.quiet:
                print(f'[PUBMED] Fetched and cached {len(papers)} papers.')

    # bioRxiv
    if args.source in ('biorxiv', 'both'):
        use_cache = False
        if not args.no_cache and storage.was_recently_fetched(search_query.query, source='biorxiv'):
            cached = storage.search_cached(search_query.query, source='biorxiv')
            if len(cached) >= search_query.max_results:
                use_cache = True
                all_papers.extend(cached[:search_query.max_results])
                if not args.quiet:
                    print(f'[CACHE] Loaded {search_query.max_results} bioRxiv papers from cache.')
            else:
                if not args.quiet:
                    print(
                        '[CACHE] bioRxiv cache has '
                        f'{len(cached)} paper(s), need {search_query.max_results}. Fetching fresh data.'
                    )

        if not use_cache:
            fetcher = BioRxivFetcher(server='biorxiv', days_back=180)
            papers = fetcher.fetch(query=search_query.query, max_results=search_query.max_results)
            storage.cache_query_results(query=search_query.query, source='biorxiv', papers=papers)
            all_papers.extend(papers)
            if not args.quiet:
                print(f'[BIORXIV] Fetched and cached {len(papers)} papers.')

    if not all_papers:
        if args.json:
            print(json.dumps({'query': search_query.query, 'results': []}, indent=2))
        elif not args.quiet:
            print('No paper found. Try a different query or check internet connection.')
        return
    
    # TF-IDF Ranking
    ranker = TFIDFRanker()
    results = ranker.rank(all_papers, query=search_query.query, top_n=10)

    if args.json:
        payload = {
            'query': search_query.query,
            'source': args.source,
            'requested_max': search_query.max_results,
            'returned': len(results),
            'results': [_result_to_dict(r) for r in results],
        }
        print(json.dumps(payload, indent=2))
        return

    if args.quiet:
        return

    print(f"\n{'='*70}")
    print(f"  TOP {len(results)} RESULTS — TF-IDF  |  Query: {search_query.query!r}")
    print(f"{'='*70}\n")

    for result in results:
        print(result)
        print()

    top_terms = ranker.get_top_terms(search_query.query, n=8)
    if top_terms:
        print(f"[DEBUG] Top TF-IDF terms for query: {top_terms}")


if __name__ == "__main__":
    main()



