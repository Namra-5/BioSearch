# main.py — Unified command-line interface for BioSearch AI.

import argparse
import json
import sys
from pathlib import Path
 
from config import DB_PATH, NCBI_API_KEY, NCBI_EMAIL, setup_logging

 
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="BioSearch AI — Biomedical literature search and evaluation system",
    )
    # Search
    p.add_argument("--query", "-q", metavar="QUERY", help="Biomedical search query")
    p.add_argument("--method", choices=["tfidf", "biobert", "both", "none"], default="tfidf")
    p.add_argument("--source", choices=["pubmed", "biorxiv", "both"], default="pubmed")
    p.add_argument("--max", "-m", type=int, default=15, metavar="N")
    # NER / graph
    p.add_argument("--ner",   action="store_true")
    p.add_argument("--graph", action="store_true")
    p.add_argument("--entity", metavar="NAME")
    p.add_argument("--save-graph",    metavar="PATH")
    p.add_argument("--save-edgelist", metavar="PATH")
    p.add_argument("--rules-only",    action="store_true")
    # Evaluation
    p.add_argument("--evaluate",   action="store_true")
    p.add_argument("--findings",   default="findings.md", metavar="PATH")
    p.add_argument("--no-biobert", action="store_true")
    # Utility
    p.add_argument("--compare",     nargs=2, metavar=("TERM_A", "TERM_B"))
    p.add_argument("--stats",       action="store_true")
    p.add_argument("--embed-stats", action="store_true")
    p.add_argument("--prune-days",  type=int, metavar="N")
    p.add_argument("--no-cache",    action="store_true")
    p.add_argument("--quiet",       action="store_true")
    p.add_argument("--json",        action="store_true")
    p.add_argument("--device",      choices=["cpu", "cuda"])
    return p
 
 
def _run_evaluation(args) -> None:
    """Empirical evaluation — TF-IDF vs BioBERT."""
    from src.fetcher_pubmed import PubMedFetcher
    from src.storage import PaperStorage
    from src.evaluator import (
        BioSearchEvaluator, save_csv, save_json_summary,
        print_report, generate_findings_md,
    )
 
    storage = PaperStorage(db_path=DB_PATH)
    if not NCBI_EMAIL:
        print('Error: Set NCBI_EMAIL in .env before running --evaluate.')
        return
    fetcher = PubMedFetcher(email=NCBI_EMAIL, api_key=NCBI_API_KEY)
    paper_cache: dict = {}
 
    def _cached_fetch(query: str, max_results: int):
        if query not in paper_cache:
            cached = storage.get_papers_for_query(query, source="pubmed")
            if cached and not args.no_cache:
                paper_cache[query] = cached
            else:
                papers = fetcher.fetch(query, max_results=max_results)
                storage.cache_query_results(query, "pubmed", papers)
                paper_cache[query] = papers
        return paper_cache[query]
 
    methods = ["tfidf"] if args.no_biobert else ["tfidf", "biobert"]
    evaluator = BioSearchEvaluator(fetcher_fn=_cached_fetch)
    run = evaluator.run(methods=methods)
 
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
 
    csv_path  = data_dir / "comparison_results.csv"
    json_path = data_dir / "findings_summary.json"
    md_path   = Path(args.findings)
 
    save_csv(run, csv_path)
    save_json_summary(run, json_path)
    generate_findings_md(run, md_path)
 
    if not args.quiet:
        print_report(run)
        print(f"\nOutputs written:")
        print(f"  {csv_path}")
        print(f"  {json_path}")
        print(f"  {md_path}")
 
 
def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
 
    if args.evaluate:
        _run_evaluation(args)
        return
 
    if args.compare:
        from src.ranker_semantic import SemanticRanker
        score = SemanticRanker().compare_texts(args.compare[0], args.compare[1])
        print(f"Similarity('{args.compare[0]}', '{args.compare[1]}') = {score:.4f}")
        return
 
    if args.stats:
        from src.storage import PaperStorage
        print(json.dumps(PaperStorage(db_path=DB_PATH).stats(), indent=2))
        return
 
    if args.embed_stats:
        from src.embedder import BioBERTEmbedder
        BioBERTEmbedder().print_cache_stats()
        return
 
    if args.prune_days:
        from src.storage import PaperStorage
        summary = PaperStorage(db_path=DB_PATH).prune_stale(older_than_days=args.prune_days)
        print(json.dumps(summary, indent=2))
        return
 
    if not args.query:
        parser.print_help()
        sys.exit(1)
 
    # Standard search mode
    from src.fetcher_pubmed  import PubMedFetcher
    from src.fetcher_biorxiv import BioRxivFetcher
    from src.storage import PaperStorage
 
    storage  = PaperStorage(db_path=DB_PATH)
    fetchers = []
    if args.source in ("pubmed",  "both"):
        if not NCBI_EMAIL:
            print('Error: Set NCBI_EMAIL in .env before using PubMed.')
            return
        fetchers.append(PubMedFetcher(email=NCBI_EMAIL, api_key=NCBI_API_KEY))
    if args.source in ("biorxiv", "both"):
        fetchers.append(BioRxivFetcher())
 
    papers = []
    for fetcher in fetchers:
        source_name = "pubmed" if fetcher.__class__.__name__ == "PubMedFetcher" else "biorxiv"
        cached = [] if args.no_cache else storage.get_papers_for_query(args.query, source=source_name)
        if cached and len(cached) >= args.max:
            papers.extend(cached[:args.max])
            continue

        fetched = fetcher.fetch(args.query, max_results=args.max)
        storage.cache_query_results(query=args.query, source=source_name, papers=fetched)
        papers.extend(fetched)
 
    if not papers:
        print("No papers found.")
        return

    if args.source == "both":
        from src.dedup import deduplicate_papers
        dedup_result = deduplicate_papers(papers)
        if not args.quiet:
            print(dedup_result.summary())
        papers = dedup_result.papers
 
    if args.method in ("tfidf", "both"):
        from src.ranker_tfidf import TFIDFRanker
        ranked = TFIDFRanker().rank(papers, args.query)
        print("\n── TF-IDF Results ──")
        for i, result in enumerate(ranked[:10], 1):
            print(f"  {i:2}. [{result.score:.4f}] {result.paper.title[:80]}")
 
    if args.method in ("biobert", "both"):
        from src.ranker_semantic import SemanticRanker
        ranked = SemanticRanker().rank(papers, args.query)
        print("\n── BioBERT Results ──")
        for i, result in enumerate(ranked[:10], 1):
            print(f"  {i:2}. [{result.score:.4f}] {result.paper.title[:80]}")
 
    if args.ner:
        from src.ner_extractor import BioNERExtractor
        ner = BioNERExtractor(use_statistical=not args.rules_only)
        ner.warm_up()
        for paper in papers[:5]:
            entities = ner.extract(paper.combined_text, paper.paper_id)
            print(f"\n[NER] {paper.title[:60]}")
            for gene in entities.genes:
                print(f"  GENE: {gene}")
            for disease in entities.diseases:
                print(f"  DISEASE: {disease}")
 
    if args.graph:
        from src.knowledge_base import KnowledgeBase
        kb    = KnowledgeBase()
        graph = kb.process_papers(papers)
        print(f"\nGraph: {graph.node_count} nodes, {graph.edge_count} edges")
        if args.entity:
            subgraph = graph.subgraph_for_entity(args.entity, depth=1)
            if subgraph.number_of_nodes() == 0:
                print(f"Entity '{args.entity}' not found in graph.")
            else:
                neighbours = graph.neighbours(args.entity)
                print(f"1-hop neighbours of '{args.entity}': {neighbours}")
        if args.save_graph:
            graph.save_json(Path(args.save_graph))
            print(f"Graph saved: {args.save_graph}")
        if args.save_edgelist:
            graph.save_edgelist(args.save_edgelist)
            print(f"Edgelist saved: {args.save_edgelist}")
 
 
if __name__ == "__main__":
    main()
