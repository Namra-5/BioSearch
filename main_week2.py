# main_week2.py
# Week 2 integration CLI — extends main_week1.py with --method flag.
# Supports: --method tfidf | biobert | both
# When 'both' is chosen, ranks with both methods and prints side-by-side for
# visual comparison

from __future__ import annotations

import argparse
import json
import sys

from config import setup_logging, NCBI_EMAIL, NCBI_API_KEY, DB_PATH
from src.models import SearchQuery
from src.fetcher_pubmed import PubMedFetcher
from src.fetcher_biorxiv import BioRxivFetcher
from src.storage import PaperStorage
from src.ranker_tfidf import TFIDFRanker
from src.ranker_semantic import SemanticRanker

def _fetch_all_papers(args: argparse.Namespace, storage: PaperStorage, 
                      search_query: SearchQuery):
    """Shared paper fetching logic."""
    all_papers = []

    if args.source in ('pubmed', 'both'):
        use_cache = False
        if not args.no_cache and storage.was_recently_fetched(search_query.query, 
                                                              source='pubmed'):
            cached = storage.search_cached(search_query.query, source='pubmed')
            if len(cached) >= search_query.max_results:
                use_cache = True
                all_papers.extend(cached[:search_query.max_results])
                if not args.quiet:
                    print(f'[CACHE] Loaded {search_query.max_results} PubMed papers from cache.')
        if not use_cache:
            fetcher = PubMedFetcher(email=NCBI_EMAIL, api_key=NCBI_API_KEY)
            papers = fetcher.fetch(query=search_query.query, 
                                   max_results=search_query.max_results)
            storage.cache_query_results(query=search_query.query, source='pubmed', 
                                        papers=papers)
            all_papers.extend(papers)
            if not args.quiet:
                print(f'[PUBMED] Fetched {len(papers)} papers.')
    
    if args.source in ('biorxiv', 'both'):
        use_cache = False
        if not args.no_cache and storage.was_recently_fetched(search_query.query, 
                                                              source='biorxiv'):
            cached = storage.search_cached(search_query.query, source='biorxiv')
            if len(cached) >= search_query.max_results:
                use_cache = True
                all_papers.extend(cached[:search_query.max_results])
                if not args.quiet:
                    print(f'[CACHE] Loaded {search_query.max_results} bioRxiv papers from cache.')
        if not use_cache:
            fetcher = BioRxivFetcher(server='biorxiv', days_back=180)
            papers = fetcher.fetch(query=search_query.query, 
                                   max_results=search_query.max_results)
            storage.cache_query_results(query=search_query.query, source='biorxiv', 
                                        papers=papers)
            all_papers.extend(papers)
            if not args.quiet:
                print(f'[BIORXIV] Fetched {len(papers)} papers.')

    return all_papers

def  _print_results(results, label: str) -> None:
    print(f'\n{'='*70}')
    print(f'  {label}')
    print(f'{'='*70}\n')
    for r in results:
        print(r)
        print()

def _print_side_by_side(tfidf_results, semantic_results, query: str) -> None:
    """Print both result sets with rank-difference annotations."""
    print(f'\n{'='*70}')
    print(f'  COMPARISON: TF-IDF vs BioBERT  |  Query: {query!r}')
    print(f'{'='*70}')

    # Build rank lookup by paper_id for each method
    tfidf_rank = {r.paper.paper_id: r.rank for r in tfidf_results}
    tfidf_score = {r.paper.paper_id: r.score for r in tfidf_results}
    
    print(f'\n{'#':>3}  {'BIOBERT':>8}  {'TF-IDF':>8}  {'ΔRANK':>6}  TITLE')
    print('-' * 70)

    for r in semantic_results:
        tf_r = tfidf_rank.get(r.paper.paper_id, 'N/A')
        delta = (tf_r - r.rank) if isinstance(tf_r, int) else '?'
        delta_str = f'+{delta}' if isinstance(delta, int) and delta > 0 else str(delta)
        title_preview = r.paper.title[:42] + '...' if len(r.paper.title) > 45 else r.paper.title
        tf_score = tfidf_score.get(r.paper.paper_id, 0.0)
        print(f'{r.rank:>3}  {r.score:>8.4f}  {tf_score:>8.4f}  {delta_str:>6}  {title_preview}')

    print(
        '\nΔRANK > 0: BioBERT promoted this paper vs TF-IDF (semantic signal found).'
        '\nΔRANK < 0: TF-IDF promoted this paper (likely keyword-heavy but semantically weak).'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='BioSearch AI — Week 2 CLI')
    parser.add_argument('--query', '-q', required=False, help='Biomedical search query')
    parser.add_argument('--max', '-m', type=int, default=10, help='Max results per source')
    parser.add_argument(
        '--source',
        choices=['pubmed', 'biorxiv', 'both'],
        default='both',
        help='Data source',
    )
    parser.add_argument(
        '--method',
        choices=['tfidf', 'biobert', 'both'],
        default='biobert',
        help='Ranking method (default: biobert)',
    )
    parser.add_argument('--no-cache', action='store_true', help='Bypass paper cache')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--embed-stats', action='store_true', help='Show embedding cache statistics')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-essential output')
    parser.add_argument('--json', action='store_true', help='Machine-readable JSON output')
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('TERM_A', 'TERM_B'),
        help='Show BioBERT cosine similarity between two terms (synonym demo)',
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default=None,
        help='Force CPU or CUDA for BioBERT (default: auto-detect)',
    )
    parser.add_argument(
        '--prune-days',
        type=int,
        metavar='DAYS',
        help='Prune paper cache rows older than DAYS and exit',
    )
    args = parser.parse_args()

    setup_logging('WARNING' if args.quiet else 'INFO')
    storage = PaperStorage(db_path=DB_PATH)

    # -- Utility sub-commands --
    if args.stats:
        print(json.dumps(storage.stats(), indent=2))
        return

    if args.prune_days is not None:
        print(json.dumps(storage.prune_stale(older_than_days=args.prune_days), indent=2))
        return

    if args.embed_stats:
        # Instantiate ranker just to read cache stats — does NOT load the model
        ranker = SemanticRanker(device=args.device)
        print(json.dumps(ranker.cache_stats(), indent=2))
        return

    if args.compare:
        # Synonym similarity demo — loads the model
        term_a, term_b = args.compare
        print(f'Loading BioBERT for synonym comparison...')
        ranker = SemanticRanker(device=args.device)
        sim = ranker.compare_texts(term_a, term_b)
        print(f'\nBioBERT cosine similarity:')
        print(f"  '{term_a}'  ↔  '{term_b}'")
        print(f'  Score: {sim:.4f}  ({"similar" if sim > 0.6 else "not similar"})')
        print(f'\nFor reference:')
        print(f'  > 0.8 = strong synonym / paraphrase')
        print(f'  0.5–0.8 = related concepts')
        print(f'  < 0.5 = semantically distant')
        return

    # Require query for all other paths 
    if not args.query:
        parser.error('--query is required unless using --stats, --embed-stats, --compare, or --prune-days')

    if args.source in ('pubmed', 'both') and not NCBI_EMAIL:
        print('Error: Set NCBI_EMAIL in .env file to use PubMed.')
        sys.exit(1)

    search_query = SearchQuery(query=args.query, max_results=args.max)
    all_papers = _fetch_all_papers(args, storage, search_query)

    if not all_papers:
        msg = 'No papers found. Try a different query or check internet connection.'
        if args.json:
            print(json.dumps({'query': args.query, 'results': []}))
        elif not args.quiet:
            print(msg)
        return

    # -- Ranking --
    top_n = 10

    if args.method == 'tfidf':
        ranker_tf = TFIDFRanker()
        results = ranker_tf.rank(all_papers, query=args.query, top_n=top_n)
        if not args.quiet:
            _print_results(results, f'TOP {len(results)} — TF-IDF | {args.query!r}')

    elif args.method == 'biobert':
        if not args.quiet:
            print('[BioBERT] Loading model (first run takes ~15s for download)...')
        ranker_sem = SemanticRanker(device=args.device)
        results = ranker_sem.rank(all_papers, query=args.query, top_n=top_n)
        if not args.quiet:
            _print_results(results, f'TOP {len(results)} — BioBERT Semantic | {args.query!r}')

    elif args.method == 'both':
        if not args.quiet:
            print('[BioBERT] Loading model...')
        ranker_tf = TFIDFRanker()
        ranker_sem = SemanticRanker(device=args.device)
        tf_results = ranker_tf.rank(all_papers, query=args.query, top_n=top_n)
        sem_results = ranker_sem.rank(all_papers, query=args.query, top_n=top_n)
        results = sem_results  # BioBERT is the primary output
        if not args.quiet:
            _print_side_by_side(tf_results, sem_results, args.query)
            tf_top = ranker_tf.get_top_terms(args.query, n=6)
            if tf_top:
                print(f'\n[TF-IDF] Top terms: {tf_top}')

    if args.json:
        # Reuse the serialisation helper pattern from main_week1.py
        def _to_dict(r):
            p = r.paper
            return {
                'rank': r.rank,
                'score': r.score,
                'method': r.method,
                'paper': {
                    'paper_id': p.paper_id,
                    'source': p.source,
                    'title': p.title,
                    'abstract': p.abstract[:200] + '...' if len(p.abstract) > 200 else p.abstract,
                    'authors': p.authors,
                    'published_date': p.published_date.isoformat() if p.published_date else None,
                    'url': p.url,
                },
            }
        payload = {
            'query': args.query,
            'method': args.method,
            'results': [_to_dict(r) for r in results],
        }
        print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
  
