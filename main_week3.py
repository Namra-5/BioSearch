# main_week3.py
# Week 3 CLI — extends Week 2 with --ner and --graph flags.
# Full backwards compatibility: all Week 1/2 flags still work.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import setup_logging, NCBI_EMAIL, NCBI_API_KEY, DB_PATH
from src.models import Paper, SearchQuery
from src.fetcher_pubmed import PubMedFetcher
from src.fetcher_biorxiv import BioRxivFetcher
from src.storage import PaperStorage
from src.ranker_tfidf import TFIDFRanker
from src.ranker_semantic import SemanticRanker
from src.knowledge_base import KnowledgeBase
from src.ner_extractor import BioNERExtractor


def _fetch_papers(args: argparse.Namespace, storage: PaperStorage, sq: SearchQuery):
    """Shared fetching logic (unchanged from Week 2)."""
    papers = []
    if args.source in ('pubmed', 'both'):
        use_cache = (
            not args.no_cache
            and storage.was_recently_fetched(sq.query, source='pubmed')
        )
        if use_cache:
            cached = storage.search_cached(sq.query, source='pubmed')
            if len(cached) >= sq.max_results:
                papers.extend(cached[: sq.max_results])
                if not args.quiet:
                    print(f'[CACHE] {len(cached[:sq.max_results])} PubMed papers from cache.')
                use_cache = True
            else:
                use_cache = False
        if not use_cache:
            fetcher = PubMedFetcher(email=NCBI_EMAIL, api_key=NCBI_API_KEY)
            fetched = fetcher.fetch(query=sq.query, max_results=sq.max_results)
            storage.cache_query_results(query=sq.query, source='pubmed', papers=fetched)
            papers.extend(fetched)
            if not args.quiet:
                print(f'[PUBMED] Fetched {len(fetched)} papers.')

    if args.source in ('biorxiv', 'both'):
        use_cache = (
            not args.no_cache
            and storage.was_recently_fetched(sq.query, source='biorxiv')
        )
        if use_cache:
            cached = storage.search_cached(sq.query, source='biorxiv')
            if len(cached) >= sq.max_results:
                papers.extend(cached[: sq.max_results])
                if not args.quiet:
                    print(f'[CACHE] {len(cached[:sq.max_results])} bioRxiv papers from cache.')
                use_cache = True
            else:
                use_cache = False
        if not use_cache:
            fetcher = BioRxivFetcher(server='biorxiv', days_back=180)
            fetched = fetcher.fetch(query=sq.query, max_results=sq.max_results)
            storage.cache_query_results(query=sq.query, source='biorxiv', papers=fetched)
            papers.extend(fetched)
            if not args.quiet:
                print(f'[BIORXIV] Fetched {len(fetched)} papers.')
    return papers


def _run_ner_only(papers: list[Paper], args: argparse.Namespace) -> None:
    """Run NER and print entity summary without building a graph."""
    extractor = BioNERExtractor(use_statistical=not args.rules_only)
    extractor.warm_up()
    extracted = extractor.extract_batch(
        [(paper.combined_text, paper.paper_id) for paper in papers],
        batch_size=64,
    )
    by_paper_id = {pe.paper_id: pe for pe in extracted}

    print(f'\n{'='*60}')
    print(f'  NER RESULTS — {len(papers)} papers')
    print(f'{'='*60}')
    for paper in papers:
        pe = by_paper_id.get(paper.paper_id)
        if pe is None:
            continue
        if pe.has_entities:
            print(f'\n[{paper.paper_id}] {paper.title[:60]}')
            if pe.genes:
                print(f'  GENES:    {', '.join(pe.genes[:8])}')
            if pe.diseases:
                print(f'  DISEASES: {', '.join(pe.diseases[:8])}')


def _run_graph(papers: list[Paper], args: argparse.Namespace) -> None:
    """Build and display knowledge graph."""
    kb = KnowledgeBase(use_statistical_ner=not args.rules_only)
    graph = kb.process_papers(papers)

    if graph.node_count == 0:
        print('\n[GRAPH] No entities found in this corpus. '
              'Try a more specific biomedical query.')
        return

    stats = graph.summary_stats(top_n=10)

    print(f'\n{'='*60}')
    print(f'  KNOWLEDGE GRAPH SUMMARY')
    print(f'{'='*60}')
    print(f'  Nodes (entities):  {stats.total_nodes}')
    print(f'  Edges (co-occur):  {stats.total_edges}')
    print(f'  Gene nodes:        {stats.gene_nodes}')
    print(f'  Disease nodes:     {stats.disease_nodes}')
    print(f'  Graph density:     {stats.density:.6f}')
    print(f'  Components:        {stats.connected_components}')

    if stats.top_genes_by_degree:
        print(f'\n  Top genes by degree centrality:')
        for name, score in stats.top_genes_by_degree[:7]:
            bar = '█' * int(score * 40)
            print(f'    {name:<18} {score:.4f}  {bar}')

    if stats.top_diseases_by_degree:
        print(f'\n  Top diseases by degree centrality:')
        for name, score in stats.top_diseases_by_degree[:7]:
            bar = '█' * int(score * 40)
            print(f'    {name:<18} {score:.4f}  {bar}')

    if stats.top_edges_by_weight:
        print(f'\n  Strongest co-occurrence edges:')
        for a, b, w in stats.top_edges_by_weight[:10]:
            print(f'    {a:<16} ↔  {b:<16}  weight={w}')

    # Gene-disease pairs only
    gd_edges = graph.gene_disease_edges()[:10]
    if gd_edges:
        print(f'\n  Top gene–disease associations:')
        for gene, disease, w in gd_edges:
            print(f'    {gene:<14} → {disease:<20}  co-occur={w}')

    # Save outputs
    if args.save_graph:
        graph.save_json(Path(args.save_graph))
        print(f'\n[GRAPH] Saved to {args.save_graph}')

    if args.save_edgelist:
        graph.save_edgelist(Path(args.save_edgelist))
        print(f'[GRAPH] Edgelist saved to {args.save_edgelist}')

    if args.json:
        print(json.dumps(stats.to_dict(), indent=2))

    # Entity-level subgraph query
    if args.entity:
        entity = args.entity.lower()
        sub = graph.subgraph_for_entity(entity, depth=1)
        print(f"\n  Subgraph for '{entity}': {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
        neighbours = graph.neighbours(entity)[:10]
        if neighbours:
            print(f'  Top co-occurring entities:')
            for nbr, w in neighbours:
                nbr_type = graph.graph.nodes[nbr].get('entity_type', '?')
                print(f'    {nbr:<20}  [{nbr_type}]  weight={w}')


def main() -> None:
    parser = argparse.ArgumentParser(description='BioSearch AI — Week 3 CLI')
    parser.add_argument('--query', '-q', help='Biomedical search query')
    parser.add_argument('--max', '-m', type=int, default=15, help='Max results per source')
    parser.add_argument('--source', choices=['pubmed', 'biorxiv', 'both'], default='pubmed')
    parser.add_argument('--method', choices=['tfidf', 'biobert', 'none'], default='tfidf',
                        help="Ranking method ('none' skips ranking — useful with --ner/--graph)")
    # Week 3 flags
    parser.add_argument('--ner', action='store_true',
                        help='Run NER and show extracted entities per paper')
    parser.add_argument('--graph', action='store_true',
                        help='Build and display knowledge graph from search results')
    parser.add_argument('--entity', type=str, default=None,
                        help='Show subgraph and neighbours for a specific entity (use with --graph)')
    parser.add_argument('--save-graph', type=str, default=None, metavar='PATH',
                        help='Save knowledge graph as JSON to PATH')
    parser.add_argument('--save-edgelist', type=str, default=None, metavar='PATH',
                        help='Save edgelist as TSV to PATH (Gephi/Cytoscape compatible)')
    parser.add_argument('--rules-only', action='store_true',
                        help='Use rules-only NER (faster, no spaCy statistical model)')
    # Shared flags
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--prune-days', type=int, metavar='DAYS')
    args = parser.parse_args()

    setup_logging('WARNING' if args.quiet else 'INFO')
    storage = PaperStorage(db_path=DB_PATH)

    if args.stats:
        print(json.dumps(storage.stats(), indent=2))
        return
    if args.prune_days:
        print(json.dumps(storage.prune_stale(older_than_days=args.prune_days), indent=2))
        return
    if not args.query:
        parser.error('--query is required')
    if args.source in ('pubmed', 'both') and not NCBI_EMAIL:
        print('Error: Set NCBI_EMAIL in .env')
        sys.exit(1)

    sq = SearchQuery(query=args.query, max_results=args.max)
    papers = _fetch_papers(args, storage, sq)

    if not papers:
        print('No papers found. Try a different query.')
        return

    # ── Ranking (optional) ─────────────────────────────────────────────────────
    if args.method == 'tfidf':
        ranker = TFIDFRanker()
        results = ranker.rank(papers, args.query, top_n=10)
        if not args.quiet and not args.ner and not args.graph:
            print(f'\n{'='*60}')
            print(f'  TOP {len(results)} — TF-IDF | {args.query!r}')
            print(f'{'='*60}\n')
            for r in results:
                print(r); print()
    elif args.method == 'biobert':
        ranker_sem = SemanticRanker()
        results = ranker_sem.rank(papers, args.query, top_n=10)
        if not args.quiet and not args.ner and not args.graph:
            print(f'\n{'='*60}')
            print(f'  TOP {len(results)} — BioBERT | {args.query!r}')
            print(f'{'='*60}\n')
            for r in results:
                print(r); print()

    # ── NER ────────────────────────────────────────────────────────────────────
    if args.ner:
        _run_ner_only(papers, args)

    # ── Graph ──────────────────────────────────────────────────────────────────
    if args.graph:
        _run_graph(papers, args)


if __name__ == '__main__':
    main()
