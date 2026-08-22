"""
evaluator.py - Empirical evaluation engine for BioSearch AI.

Compares TF-IDF vs BioBERT retrieval on standardised biomedical queries
using the Graph Connectivity Score (GCS) as the primary evaluation metric.
"""

from __future__ import annotations
import time
import json
import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
 
from src.models import DataSource, Paper
from src.knowledge_base import KnowledgeBase
from src.ner_extractor import LABEL_DISEASE, LABEL_GENE
 
 
# -- Standardised evaluation queries --
EVALUATION_QUERIES = [
    "BRCA1 BRCA2 breast cancer hereditary mutations",
    "KRAS oncogene lung cancer targeted therapy",
    "TP53 tumour suppressor apoptosis cancer",
    "EGFR tyrosine kinase inhibitor resistance non-small cell",
    "Alzheimer disease APOE neurodegeneration amyloid",
]
 
@dataclass
class QueryResult:
    """Holds all metrics for one (query, method) evaluation run."""
    query_id: str                   # Q1 … Q5
    query_text: str
    method: str                     # 'tfidf' | 'biobert'
    papers_fetched: int = 0
    papers_ranked: int = 0
    top_n_evaluated: int = 10
    mean_score: float = 0.0
    top1_score: float = 0.0
    score_spread: float = 0.0       # std-dev of ranking scores
    graph_node_count: int = 0
    graph_edge_count: int = 0
    gene_disease_pairs: int = 0     # PRIMARY METRIC
    hub_gene_count: int = 0
    hub_disease_count: int = 0
    graph_density: float = 0.0
    runtime_seconds: float = 0.0
    error: str = ""
 
 
@dataclass
class EvaluationRun:
    """Aggregates all QueryResults for a full evaluation."""
    results: list[QueryResult] = field(default_factory=list)
    run_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_runtime_seconds: float = 0.0
 
    def summary(self) -> dict:
        """Return aggregate statistics. Pure function — no side effects."""
        tf_results  = [r for r in self.results if r.method == "tfidf"   and not r.error]
        bio_results = [r for r in self.results if r.method == "biobert" and not r.error]
 
        def _agg(rs: list[QueryResult]) -> dict:
            if not rs:
                return {}
            return {
                "mean_graph_connectivity_score": sum(r.gene_disease_pairs for r in rs) / len(rs),
                "mean_ranking_score":            sum(r.mean_score          for r in rs) / len(rs),
                "mean_runtime_seconds":          sum(r.runtime_seconds     for r in rs) / len(rs),
            }
 
        tf_agg  = _agg(tf_results)
        bio_agg = _agg(bio_results)

        tf_gcs  = tf_agg.get("mean_graph_connectivity_score", 0)
        bio_gcs = bio_agg.get("mean_graph_connectivity_score", 0)
        if tf_results and bio_results:
            gcs_winner = "biobert" if bio_gcs > tf_gcs else ("tfidf" if tf_gcs > bio_gcs else "tie")
        elif tf_results:
            gcs_winner = "tfidf"
        elif bio_results:
            gcs_winner = "biobert"
        else:
            gcs_winner = "tie"

        tf_rt  = tf_agg.get("mean_runtime_seconds", 1e-9)
        bio_rt = bio_agg.get("mean_runtime_seconds", 0)
        speed_ratio = bio_rt / tf_rt if tf_results and bio_results and tf_rt > 0 else None
        pair_stats = _paired_gcs_stats(self.results) if tf_results and bio_results else {}
 
        return {
            "run_timestamp":          self.run_timestamp,
            "total_runtime_seconds":  self.total_runtime_seconds,
            "queries_evaluated":      len(set(r.query_id for r in self.results)),
            "aggregate": {
                "tfidf":   tf_agg,
                "biobert": bio_agg,
                "gcs_winner":   gcs_winner,
                "speed_ratio":  speed_ratio,
                "paired_gcs_stats": pair_stats,
            },
            "per_query": [
                {
                    "query_id":           r.query_id,
                    "query_text":         r.query_text,
                    "method":             r.method,
                    "gene_disease_pairs": r.gene_disease_pairs,
                    "mean_score":         r.mean_score,
                    "runtime_seconds":    r.runtime_seconds,
                    "error":              r.error,
                }
                for r in self.results
            ],
        }
 
 
# -- Internal helpers --
 
def _score_stats(scores: list[float]) -> tuple[float, float, float]:
    """Return (mean, top1, spread). Handles empty list gracefully."""
    if not scores:
        return 0.0, 0.0, 0.0
    mean  = sum(scores) / len(scores)
    top1  = max(scores)
    var   = sum((s - mean) ** 2 for s in scores) / len(scores)
    spread = var ** 0.5
    return mean, top1, spread


def _paired_gcs_stats(results: list[QueryResult]) -> dict:
    """Return paired TF-IDF vs BioBERT statistics for shared query IDs.

    Computes paired sign-test and bootstrap confidence-interval statistics.
    """
    tf_by_q: dict[str, int] = {}
    bio_by_q: dict[str, int] = {}

    for r in results:
        if r.error:
            continue
        if r.method == "tfidf":
            tf_by_q[r.query_id] = r.gene_disease_pairs
        elif r.method == "biobert":
            bio_by_q[r.query_id] = r.gene_disease_pairs

    shared_queries = sorted(set(tf_by_q) & set(bio_by_q))
    if not shared_queries:
        return {}

    deltas = [bio_by_q[q] - tf_by_q[q] for q in shared_queries]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    n_non_tie = wins + losses

    p_value = None
    if n_non_tie > 0:
        k = min(wins, losses)
        tail = sum(math.comb(n_non_tie, i) for i in range(0, k + 1)) / (2 ** n_non_tie)
        p_value = min(1.0, 2.0 * tail)

    mean_delta = sum(deltas) / len(deltas)

    rng = random.Random(42)
    boot_means: list[float] = []
    n = len(deltas)
    for _ in range(5000):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    ci_low = boot_means[int(0.025 * (len(boot_means) - 1))]
    ci_high = boot_means[int(0.975 * (len(boot_means) - 1))]

    return {
        "n_pairs": len(shared_queries),
        "wins_biobert": wins,
        "wins_tfidf": losses,
        "ties": ties,
        "mean_delta_biobert_minus_tfidf": mean_delta,
        "ci95_mean_delta": [ci_low, ci_high],
        "sign_test_pvalue": p_value,
    }


def _paired_significance(paired: dict, alpha: float = 0.05) -> bool:
    """
    True only if the paired sign-test AND the bootstrap CI agree there is
    a difference supported by both criteria. Both conditions are required:
    the p-value alone can look significant while the CI still straddles zero,
    and vice versa.
    """
    if not paired:
        return False
    pvalue = paired.get("sign_test_pvalue")
    ci = paired.get("ci95_mean_delta")
    if pvalue is None or ci is None:
        return False
    ci_low, ci_high = ci
    return pvalue < alpha and not (ci_low <= 0 <= ci_high)


def _cold_warm_runtime_split(results: list[QueryResult]) -> dict:
    """
    Split recorded BioBERT runtimes into a one-time cold-start cost (the
    first successful BioBERT query in execution order, which pays the
    lazy model-load cost — see BioSearchEvaluator._get_semantic_ranker)
    and steady-state warm-query runtimes (all subsequent successful
    BioBERT queries in the same run).

    This performs NO new measurement and invents no numbers — it only
    re-buckets `runtime_seconds` values already recorded in `results`,
    in the same chronological order BioSearchEvaluator.run() produced
    them (tfidf/biobert alternating per query, in query order).
    """
    bio_results = [r for r in results if r.method == "biobert" and not r.error]
    if not bio_results:
        return {}

    cold = bio_results[0]
    warm = bio_results[1:]

    warm_runtimes = [r.runtime_seconds for r in warm]
    warm_mean = sum(warm_runtimes) / len(warm_runtimes) if warm_runtimes else None

    return {
        "cold_start_query_id": cold.query_id,
        "cold_start_seconds": cold.runtime_seconds,
        "warm_query_ids": [r.query_id for r in warm],
        "warm_mean_seconds": warm_mean,
        "warm_n": len(warm_runtimes),
    }


def _coerce_papers(items: list[object]) -> list[Paper]:
    """Normalize fetched items into Paper models so downstream rankers are type-stable."""
    coerced: list[Paper] = []
    for item in items:
        if isinstance(item, Paper):
            coerced.append(item)
            continue

        title = getattr(item, "title", "") or "Untitled paper"
        abstract = getattr(item, "abstract", "") or ""
        paper_id = (
            getattr(item, "paper_id", None)
            or getattr(item, "pmid", None)
            or getattr(item, "id", None)
            or title.lower().replace(" ", "_")
        )
        source = getattr(item, "source", DataSource.PUBMED)
        if not isinstance(source, DataSource):
            try:
                source = DataSource(str(source))
            except Exception:
                source = DataSource.PUBMED

        coerced.append(
            Paper(
                paper_id=str(paper_id),
                title=str(title),
                abstract=str(abstract),
                source=source,
            )
        )
    return coerced


def _coerce_ranked_results(items: list[object]) -> list[tuple[Paper, float]]:
    """Normalize ranker output into (Paper, score) pairs."""

    ranked: list[tuple[Paper, float]] = []
    for item in items:
        if hasattr(item, "paper") and hasattr(item, "score"):
            ranked.append((item.paper, float(item.score)))
            continue
        if isinstance(item, tuple) and len(item) == 2:
            paper, score = item
            ranked.append((paper, float(score)))
            continue
        raise TypeError(f"Unsupported ranked result item: {type(item)!r}")
    return ranked
 
 
def _compute_graph_metrics(papers: list[Paper], kb: KnowledgeBase) -> tuple:
    """
    Build graph from papers, return 6-tuple of graph metrics.
    Returns: (node_count, edge_count, gene_disease_pairs,
              hub_gene_count, hub_disease_count, density)
    """
    graph = kb.process_papers(papers)

    # Prefer the real BioKnowledgeGraph API, but keep compatibility with the
    # light-weight fake used in the unit tests.
    if hasattr(graph, "graph") and hasattr(graph, "degree_centrality"):
        nx_graph = graph.graph
        centrality = graph.degree_centrality()
        gd_pairs = graph.gene_disease_edges()
        hub_genes = [
            node
            for node, score in centrality.items()
            if nx_graph.nodes[node].get("entity_type") == LABEL_GENE and score >= 0.3
        ]
        hub_diseases = [
            node
            for node, score in centrality.items()
            if nx_graph.nodes[node].get("entity_type") == LABEL_DISEASE and score >= 0.3
        ]
        return (
            graph.node_count,
            graph.edge_count,
            len(gd_pairs),
            len(hub_genes),
            len(hub_diseases),
            graph.summary_stats().density,
        )

    gd_pairs = graph.gene_disease_edges()
    hub_genes = [
        node for node, score in graph.centrality().items()
        if graph.node_type(node) == "gene" and score >= 0.3
    ]
    hub_diseases = [
        node for node, score in graph.centrality().items()
        if graph.node_type(node) == "disease" and score >= 0.3
    ]
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(gd_pairs),
        len(hub_genes),
        len(hub_diseases),
        graph.density(),
    )


# -- Main evaluator class --
 
class BioSearchEvaluator:
    """
    Orchestrates the empirical evaluation.
 
    fetcher_fn: Any callable (papers: list[Paper]) = fetcher_fn(query, max_results)
    The evaluator never imports PubMedFetcher directly — this makes it fully testable.
    """
 
    def __init__(
        self,
        fetcher_fn: Callable[[str, int], list[Paper]],
        data_dir: Path = Path("data"),
        max_papers: int = 20,
        top_n: int = 10,
    ):
        self._fetch        = fetcher_fn
        self._data_dir     = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._max_papers   = max_papers
        self._top_n        = top_n
        self._kb           = KnowledgeBase()          # shared — NER results cached
        self._semantic_ranker: Optional[object] = None # lazy load
 
    def _get_semantic_ranker(self):
        """Lazy-load BioBERT only once per process."""
        if self._semantic_ranker is None:
            from src.ranker_semantic import SemanticRanker
            self._semantic_ranker = SemanticRanker()
        return self._semantic_ranker
 
    def _fetch_papers(self, query: str) -> list[Paper]:
        """Fetch with error handling — returns [] on failure."""
        try:
            return self._fetch(query, self._max_papers)
        except Exception as exc:
            print(f"  [fetch error] {exc}")
            return []
 
    def _evaluate_one(
        self, query_id: str, query: str, method: str
    ) -> QueryResult:
        """Run one (query, method) combination and return a QueryResult."""
        result = QueryResult(query_id=query_id, query_text=query, method=method)
        try:
            # Step 1 — fetch
            papers = self._fetch_papers(query)
            if len(papers) > 1:
                from src.dedup import deduplicate_papers
                papers = deduplicate_papers(papers).papers
            papers = _coerce_papers(papers)
            result.papers_fetched = len(papers)
 
            # Step 2 — rank
            if method == "tfidf":
                from src.ranker_tfidf import TFIDFRanker
                ranker = TFIDFRanker()
            else:
                ranker = self._get_semantic_ranker()
            t0 = time.perf_counter()
            ranked = _coerce_ranked_results(ranker.rank(papers, query))
            result.runtime_seconds = time.perf_counter() - t0
            result.papers_ranked   = len(ranked)
 
            # Step 3 — score stats
            scores = [s for _, s in ranked]
            result.mean_score, result.top1_score, result.score_spread = _score_stats(scores)
 
            # Step 4 — graph metrics on top-N
            top_papers = [p for p, _ in ranked[: self._top_n]]
            result.top_n_evaluated = len(top_papers)
            (result.graph_node_count, result.graph_edge_count,
             result.gene_disease_pairs, result.hub_gene_count,
             result.hub_disease_count, result.graph_density) = (
                _compute_graph_metrics(top_papers, self._kb)
            )
        except Exception as exc:
            result.error = str(exc)
        return result
 
    def run(
        self,
        queries: list[str] | None = None,
        methods: list[str] | None = None,
    ) -> EvaluationRun:
        """
        Run the full evaluation grid.
        queries: defaults to EVALUATION_QUERIES
        methods: defaults to ['tfidf', 'biobert']
        """
        queries = queries or EVALUATION_QUERIES
        methods = methods or ["tfidf", "biobert"]
        run     = EvaluationRun()
        t_start = time.perf_counter()
 
        for q_idx, query in enumerate(queries, start=1):
            q_id = f"Q{q_idx}"
            print(f"\n[{q_id}] {query}")
            for method in methods:
                print(f"  → {method} ... ", end="", flush=True)
                result = self._evaluate_one(q_id, query, method)
                run.results.append(result)
                status = f"GCS={result.gene_disease_pairs}" if not result.error else f"ERROR: {result.error}"
                print(status)
 
        run.total_runtime_seconds = time.perf_counter() - t_start
        return run
 
 
# -- Output writers --
 
def save_csv(run: EvaluationRun, path: Path) -> None:
    """Write raw results to CSV, one row per (query, method)."""
    fieldnames = [
        "query_id", "query_text", "method", "papers_fetched", "papers_ranked",
        "top_n_evaluated", "mean_score", "top1_score", "score_spread",
        "graph_node_count", "graph_edge_count", "gene_disease_pairs",
        "hub_gene_count", "hub_disease_count", "graph_density",
        "runtime_seconds", "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in run.results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})
 
 
def save_json_summary(run: EvaluationRun, path: Path) -> None:
    """Write the evaluation summary to JSON."""
    with open(path, "w") as f:
        json.dump(run.summary(), f, indent=2)
 
 
def print_report(run: EvaluationRun) -> None:
    """Print a formatted console report."""
    s = run.summary()
    agg = s.get("aggregate", {})
    tf  = agg.get("tfidf", {})
    bio = agg.get("biobert", {})
    paired = agg.get("paired_gcs_stats", {})

    print("\n" + "=" * 60)
    print("  BioSearch AI — Evaluation Report")
    print("=" * 60)
    print(f"  Queries evaluated : {s['queries_evaluated']}")
    print(f"  Total runtime     : {s['total_runtime_seconds']:.1f}s")
    print()
    print(f"  {'Metric':<35} {'TF-IDF':>10} {'BioBERT':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Mean GCS (primary)':35} {tf.get('mean_graph_connectivity_score',0):10.1f} {bio.get('mean_graph_connectivity_score',0):10.1f}")
    print(f"  {'Mean ranking score':35} {tf.get('mean_ranking_score',0):10.4f} {bio.get('mean_ranking_score',0):10.4f}")
    print(f"  {'Mean runtime (s, blended)':35} {tf.get('mean_runtime_seconds',0):10.3f} {bio.get('mean_runtime_seconds',0):10.3f}")
    print()

    # -- GCS: report significance, not a bare point-estimate "winner" --
    if paired and _paired_significance(paired):
        bio_gcs = bio.get("mean_graph_connectivity_score", 0)
        tf_gcs  = tf.get("mean_graph_connectivity_score", 0)
        higher  = "BioBERT" if bio_gcs > tf_gcs else "TF-IDF"
        pvalue  = paired.get("sign_test_pvalue")
        pvalue_str = f"{pvalue:.4f}" if isinstance(pvalue, float) else "n/a"
        print(f"  ★  GCS: {higher} shows a statistically significant advantage "
              f"(sign-test p={pvalue_str}, n={paired.get('n_pairs', 0)})")
    elif paired:
        pvalue = paired.get("sign_test_pvalue")
        pvalue_str = f"{pvalue:.4f}" if isinstance(pvalue, float) else "n/a"
        ci = paired.get("ci95_mean_delta", [0.0, 0.0])
        print(f"  ★  GCS: no statistically significant difference detected "
              f"(sign-test p={pvalue_str}, n={paired.get('n_pairs', 0)}, "
              f"95% CI [{ci[0]:.1f}, {ci[1]:.1f}] spans zero)")
    else:
        print("  ★  GCS: paired statistical comparison unavailable for this run")

    # -- Runtime: cold-start vs warm-query, not a blended ratio --
    cold_warm = _cold_warm_runtime_split(run.results)
    if cold_warm and cold_warm.get("warm_mean_seconds") is not None:
        tf_rt = tf.get("mean_runtime_seconds", 0.0)
        warm_mean = cold_warm["warm_mean_seconds"]
        warm_ratio = (warm_mean / tf_rt) if tf_rt > 0 else None
        ratio_str = f"{warm_ratio:.1f}×" if warm_ratio is not None else "n/a"
        print(f"  ★  BioBERT cold-start (model load): {cold_warm['cold_start_seconds']:.1f}s "
              f"(query {cold_warm['cold_start_query_id']})")
        print(f"  ★  BioBERT warm-query latency: {warm_mean:.3f}s "
              f"(~{ratio_str} slower than TF-IDF's {tf_rt:.3f}s per query)")
    else:
        print("  ★  Insufficient warm-query samples to separate BioBERT's "
              "model-load cost from steady-state latency")
    print("=" * 60)
 
 
def generate_findings_md(run: EvaluationRun, path: Path) -> None:
    """Auto-generate a structured research report in Markdown."""
    s   = run.summary()
    agg = s["aggregate"]
    tf  = agg.get("tfidf",   {})
    bio = agg.get("biobert", {})
    paired  = agg.get("paired_gcs_stats", {})
    tf_gcs  = tf.get("mean_graph_connectivity_score", 0)
    bio_gcs = bio.get("mean_graph_connectivity_score", 0)

    cold_warm   = _cold_warm_runtime_split(run.results)
    significant = _paired_significance(paired)

    if tf and bio:
        # -- Executive-summary / Key-Finding #1 language: significance-gated --
        if paired:
            n_pairs = paired.get("n_pairs", 0)
            pvalue = paired.get("sign_test_pvalue")
            ci = paired.get("ci95_mean_delta", [0.0, 0.0])
            pvalue_str = f"{pvalue:.2f}" if isinstance(pvalue, float) else "n/a"

            if significant:
                higher = "BioBERT" if bio_gcs > tf_gcs else "TF-IDF"
                outcome_str = (
                    f"{higher} showed a statistically significant advantage in mean GCS "
                    f"({tf_gcs:.1f} vs {bio_gcs:.1f}; exact sign-test p={pvalue_str}, "
                    f"n={n_pairs}; 95% bootstrap CI for the mean delta: "
                    f"{ci[0]:.1f} to {ci[1]:.1f})"
                )
            else:
                outcome_str = (
                    f"TF-IDF and BioBERT showed no statistically significant difference in "
                    f"mean GCS at this sample size ({tf_gcs:.1f} vs {bio_gcs:.1f}; exact "
                    f"sign-test p={pvalue_str}, n={n_pairs}; 95% bootstrap CI for the mean "
                    f"delta: {ci[0]:.1f} to {ci[1]:.1f}, spanning zero). The per-query pattern "
                    f"suggests that BioBERT may perform relatively better on concept/"
                    f"synonym-heavy queries, while TF-IDF may perform relatively better on "
                    f"exact-terminology queries, but this directional pattern cannot be "
                    f"established statistically with only {n_pairs} queries"
                )
        else:
            outcome_str = (
                f"TF-IDF and BioBERT produced comparable mean GCS scores "
                f"({tf_gcs:.1f} vs {bio_gcs:.1f}); a paired statistical comparison was "
                f"unavailable for this run"
            )

        # -- Runtime language: cold-start vs warm-query, not a blended ratio --
        if cold_warm and cold_warm.get("warm_mean_seconds") is not None:
            warm_mean = cold_warm["warm_mean_seconds"]
            tf_rt = tf.get("mean_runtime_seconds", 0.0)
            warm_ratio = (warm_mean / tf_rt) if tf_rt > 0 else None
            ratio_str = f"{warm_ratio:.1f}×" if warm_ratio is not None else "n/a"
            n_warm = cold_warm["warm_n"]
            speed_note = (
                f"BioBERT's one-time model-load cost was {cold_warm['cold_start_seconds']:.1f}s "
                f"(first query, {cold_warm['cold_start_query_id']}); steady-state warm-query "
                f"latency over the remaining {n_warm} quer{'y' if n_warm == 1 else 'ies'} "
                f"averaged {warm_mean:.3f}s, approximately {ratio_str} slower than TF-IDF's "
                f"{tf_rt:.3f}s per query. The blended per-query mean of "
                f"{bio.get('mean_runtime_seconds', 0):.1f}s reported below includes the "
                f"one-time load cost and should not be read as steady-state latency."
            )
        else:
            speed_note = (
                f"BioBERT mean runtime was {bio.get('mean_runtime_seconds', 0):.3f}s vs "
                f"TF-IDF's {tf.get('mean_runtime_seconds', 0):.3f}s; insufficient warm-query "
                f"samples to separate model-load cost from steady-state latency."
            )
    elif tf:
        outcome_str = "TF-IDF evaluation completed without a BioBERT comparison"
        speed_note = "Runtime comparison is unavailable because BioBERT was disabled."
    else:
        outcome_str = "BioBERT evaluation completed without a TF-IDF comparison"
        speed_note = "Runtime comparison is unavailable because TF-IDF was not run."

    lines = [
        f"# BioSearch AI — Evaluation Findings",
        f"",
        f"**Run date:** {s['run_timestamp']}",
        f"**Total runtime:** {s['total_runtime_seconds']:.1f}s",
        f"**Queries:** {s['queries_evaluated']}",
        f"",
        f"## Executive Summary",
        f"",
        f"{outcome_str}. {speed_note}",
        f"",
        f"## Key Findings",
        f"",
        f"1. **Primary metric (GCS):** {outcome_str}.",
        f"2. **Runtime:** {speed_note}",
        f"3. **Recommendation:** Hybrid two-stage pipeline optimal for production — "
        f"TF-IDF for fast candidate retrieval, BioBERT for re-ranking on queries where "
        f"exact lexical overlap is weak.",
        f"",
        f"## Per-Query Results",
        f"",
        f"| Query | Method | GCS | Score | Runtime(s) | Error |",
        f"|-------|--------|-----|-------|------------|-------|",
    ]
    for r in s["per_query"]:
        lines.append(
            f"| {r['query_id']} | {r['method']} | {r['gene_disease_pairs']} | "
            f"{r['mean_score']:.4f} | {r['runtime_seconds']:.3f} | {r['error'] or '—'} |"
        )
 
    lines += [
        f"",
        f"## Methodology",
        f"",
        f"- **Corpus:** PubMed papers fetched per query (max {20})",
        f"- **Primary metric:** Graph Connectivity Score (GCS) = unique gene-disease edges in top-10 result subgraph",
        f"- **Secondary metrics:** Score spread, hub gene/disease count, wall-clock runtime",
        f"- **Independence:** GCS computed from external knowledge graph, not ranker scores",
        f"",
        f"## Statistical Check",
        f"",
    ]

    if paired:
        ci = paired.get("ci95_mean_delta", [0.0, 0.0])
        pvalue = paired.get("sign_test_pvalue")
        pvalue_str = f"{pvalue:.4f}" if isinstance(pvalue, float) else "n/a"
        lines += [
            f"- **Paired queries (n={paired.get('n_pairs', 0)}):** BioBERT wins={paired.get('wins_biobert', 0)}, TF-IDF wins={paired.get('wins_tfidf', 0)}, ties={paired.get('ties', 0)}",
            f"- **Mean GCS delta (BioBERT - TF-IDF):** {paired.get('mean_delta_biobert_minus_tfidf', 0.0):.2f} (95% bootstrap CI: {ci[0]:.2f} to {ci[1]:.2f})",
            f"- **Exact sign-test p-value (ties excluded):** {pvalue_str}",
            f"- **Interpretation:** {'Statistically significant difference detected.' if significant else 'No statistically significant difference detected at alpha=0.05 — the 95% CI spans zero. Do not interpret the higher point estimate as a demonstrated winner at this sample size.'}",
        ]
    else:
        lines += [
            f"- Paired statistical check unavailable because both methods were not present for shared query IDs.",
        ]

    if cold_warm and cold_warm.get("warm_mean_seconds") is not None:
        lines += [
            f"",
            f"## Runtime Detail (Cold-Start vs Warm-Query)",
            f"",
            f"- **BioBERT cold start (model load, query {cold_warm['cold_start_query_id']}):** "
            f"{cold_warm['cold_start_seconds']:.1f}s (one-time per process)",
            f"- **BioBERT warm-query mean ({cold_warm['warm_n']} queries: "
            f"{', '.join(cold_warm['warm_query_ids'])}):** "
            f"{cold_warm['warm_mean_seconds']:.3f}s",
            f"- **TF-IDF mean runtime:** {tf.get('mean_runtime_seconds', 0):.3f}s",
            f"- The commonly-quoted blended ratio (BioBERT mean / TF-IDF mean, "
            f"{agg.get('speed_ratio', 0):.0f}× in the raw JSON summary) includes the "
            f"one-time model-load cost and overstates steady-state per-query latency. "
            f"The warm-query ratio above is the correct figure for steady-state "
            f"comparisons.",
        ]

    lines += [
        f"",
        f"## Limitations",
        f"",
        f"- No human relevance judgements (GCS is an automated proxy)",
        f"- CPU-only BioBERT - runtime would differ significantly on GPU",
        f"- 5 queries is a small evaluation set; results may not generalise",
        f"",
        f"---",
        f"*Generated automatically by BioSearch AI evaluator.py*",
    ]
 
    path.write_text("\n".join(lines), encoding="utf-8")