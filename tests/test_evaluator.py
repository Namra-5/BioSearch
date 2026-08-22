# tests/test_evaluator.py
 
from __future__ import annotations
import json
import math
import pytest
from dataclasses import asdict
from pathlib import Path
from typing import Optional
 
# -- Fake objects --
 
class _FakePaper:
    def __init__(self, title: str, abstract: str = ""):
        self.title    = title
        self.abstract = abstract
        self.pmid     = title.lower().replace(" ", "_")
 
FAKE_PAPERS = [
    _FakePaper("BRCA1 role in breast cancer", "BRCA1 mutations increase breast cancer risk."),
    _FakePaper("TP53 and apoptosis",           "TP53 regulates apoptosis in tumour cells."),
    _FakePaper("KRAS in lung cancer",          "KRAS mutations found in lung carcinoma."),
    _FakePaper("EGFR resistance mechanisms",   "EGFR T790M mutation causes resistance."),
    _FakePaper("Alzheimer APOE review",        "APOE4 allele increases Alzheimer risk."),
    _FakePaper("BRCA2 ovarian cancer",         "BRCA2 linked to hereditary ovarian cancer."),
    _FakePaper("TP53 colorectal",              "TP53 mutations common in colorectal cancer."),
    _FakePaper("EGFR NSCLC therapy",           "NSCLC responds to EGFR inhibitors."),
    _FakePaper("Amyloid plaques Alzheimer",    "Beta-amyloid plaques are Alzheimer hallmark."),
    _FakePaper("KRAS colon cancer",            "KRAS mutations in colon adenocarcinoma."),
]
 
def _fake_fetcher(query: str, max_results: int) -> list:
    return FAKE_PAPERS
 
class _FakeTFIDFRanker:
    """Returns deterministic descending scores, no sklearn."""
    def rank(self, papers: list, query: str) -> list:
        return [(p, 1.0 - 0.05 * i) for i, p in enumerate(papers)]
 
class _FakeSemanticRanker:
    """Returns deterministic ascending scores (reversed order)."""
    def rank(self, papers: list, query: str) -> list:
        return [(p, 0.5 + 0.03 * i) for i, p in enumerate(reversed(papers))]
 
class _FakeGraph:
    """Pre-built graph with known, deterministic structure."""
    def gene_disease_edges(self):
        return [("BRCA1","breast_cancer"), ("TP53","colorectal"), ("KRAS","lung_cancer"),
                ("EGFR","nsclc"), ("APOE","alzheimer")]
    def centrality(self):
        return {"BRCA1": 0.4, "TP53": 0.35, "breast_cancer": 0.31,
                "lung_cancer": 0.1, "KRAS": 0.05}
    def node_type(self, n):
        return "disease" if n in ("breast_cancer","colorectal","lung_cancer","nsclc","alzheimer") else "gene"
    def number_of_nodes(self): return 10
    def number_of_edges(self): return 8
    def density(self): return 0.42
 
class _FakeKnowledgeBase:
    def process_papers(self, papers): return _FakeGraph()
 
def _make_evaluator(tmp_path: Path):
    """Create a BioSearchEvaluator with all fakes injected."""
    from src.evaluator import BioSearchEvaluator
    ev = BioSearchEvaluator(fetcher_fn=_fake_fetcher, data_dir=tmp_path)
    ev._kb = _FakeKnowledgeBase()
    return ev
 
# -- _score_stats --
 
class TestScoreStats:
    def test_basic(self):
        from src.evaluator import _score_stats
        mean, top1, spread = _score_stats([0.9, 0.7, 0.5, 0.3])
        assert abs(mean - 0.6) < 1e-9
        assert top1 == 0.9
        assert spread > 0
 
    def test_single_element(self):
        from src.evaluator import _score_stats
        mean, top1, spread = _score_stats([0.8])
        assert mean == 0.8
        assert top1 == 0.8
        assert spread == 0.0
 
    def test_empty(self):
        from src.evaluator import _score_stats
        mean, top1, spread = _score_stats([])
        assert mean == top1 == spread == 0.0
 
    def test_uniform_scores(self):
        from src.evaluator import _score_stats
        mean, top1, spread = _score_stats([0.5, 0.5, 0.5])
        assert mean == 0.5
        assert spread == 0.0
 
    def test_spread_correct(self):
        from src.evaluator import _score_stats
        _, _, spread = _score_stats([1.0, 0.0])
        assert abs(spread - 0.5) < 1e-9
 
# -- QueryResult --
 
class TestQueryResult:
    def test_defaults(self):
        from src.evaluator import QueryResult
        r = QueryResult(query_id="Q1", query_text="test", method="tfidf")
        assert r.gene_disease_pairs == 0
        assert r.error == ""
        assert r.top_n_evaluated == 10
 
    def test_fields_settable(self):
        from src.evaluator import QueryResult
        r = QueryResult(query_id="Q2", query_text="KRAS", method="biobert",
                        gene_disease_pairs=7, runtime_seconds=12.3)
        assert r.gene_disease_pairs == 7
        assert r.runtime_seconds == 12.3
 
# -- EvaluationRun.summary() --
 
class TestEvaluationRunSummary:
    def _make_run(self, tf_gcs, bio_gcs):
        from src.evaluator import EvaluationRun, QueryResult
        run = EvaluationRun()
        run.results = [
            QueryResult("Q1","q","tfidf",   gene_disease_pairs=tf_gcs,  runtime_seconds=0.04),
            QueryResult("Q1","q","biobert", gene_disease_pairs=bio_gcs, runtime_seconds=10.0),
        ]
        return run
 
    def test_biobert_wins(self):
        run = self._make_run(3, 5)
        s   = run.summary()
        assert s["aggregate"]["gcs_winner"] == "biobert"
 
    def test_tfidf_wins(self):
        run = self._make_run(7, 3)
        s   = run.summary()
        assert s["aggregate"]["gcs_winner"] == "tfidf"
 
    def test_tie(self):
        run = self._make_run(4, 4)
        s   = run.summary()
        assert s["aggregate"]["gcs_winner"] == "tie"
 
    def test_speed_ratio(self):
        run = self._make_run(3, 5)
        s   = run.summary()
        assert abs(s["aggregate"]["speed_ratio"] - 250.0) < 1.0  # 10/0.04
 
    def test_per_query_list(self):
        run = self._make_run(3, 5)
        s   = run.summary()
        assert len(s["per_query"]) == 2

    def test_paired_gcs_stats_present(self):
        run = self._make_run(3, 5)
        s = run.summary()
        paired = s["aggregate"]["paired_gcs_stats"]
        assert paired["n_pairs"] == 1
        assert paired["wins_biobert"] == 1
        assert paired["wins_tfidf"] == 0
        assert "ci95_mean_delta" in paired
 
    def test_errors_excluded_from_aggregate(self):
        from src.evaluator import EvaluationRun, QueryResult
        run = EvaluationRun()
        run.results = [
            QueryResult("Q1","q","tfidf",   gene_disease_pairs=6, runtime_seconds=0.04),
            QueryResult("Q1","q","biobert", gene_disease_pairs=0, runtime_seconds=0.0, error="CUDA OOM"),
        ]
        s = run.summary()
        assert s["aggregate"]["biobert"].get("mean_graph_connectivity_score", None) is None or \
               s["aggregate"]["gcs_winner"] == "tfidf"
 
# -- BioSearchEvaluator.run() --
 
class TestEvaluatorRun:
    def test_run_tfidf_single_query(self, tmp_path):
        ev = _make_evaluator(tmp_path)
        # Inject fake tfidf ranker
        import src.evaluator as ev_mod
        orig = None
        try:
            from src import ranker_tfidf
            orig = ranker_tfidf.TFIDFRanker
            ranker_tfidf.TFIDFRanker = _FakeTFIDFRanker
        except Exception:
            pass
 
        run = ev.run(queries=["BRCA1 breast cancer"], methods=["tfidf"])
        assert len(run.results) == 1
        assert run.results[0].method == "tfidf"
        assert run.results[0].error == ""
 
        if orig:
            ranker_tfidf.TFIDFRanker = orig
 
    def test_run_two_methods_two_queries(self, tmp_path):
        ev  = _make_evaluator(tmp_path)
        run = ev.run(queries=["Q_A", "Q_B"], methods=["tfidf"])
        # 2 queries × 1 method = 2 results
        assert len(run.results) == 2
 
    def test_gcs_populated(self, tmp_path):
        ev  = _make_evaluator(tmp_path)
        run = ev.run(queries=["BRCA1"], methods=["tfidf"])
        r   = run.results[0]
        assert r.gene_disease_pairs == 5
 
    def test_hub_counts(self, tmp_path):
        ev  = _make_evaluator(tmp_path)
        run = ev.run(queries=["test"], methods=["tfidf"])
        r   = run.results[0]
        assert r.hub_gene_count == 2
        assert r.hub_disease_count == 1
 
    def test_fetch_error_returns_empty(self, tmp_path):
        from src.evaluator import BioSearchEvaluator
        def bad_fetch(q, n): raise ConnectionError("timeout")
        ev = BioSearchEvaluator(fetcher_fn=bad_fetch, data_dir=tmp_path)
        ev._kb = _FakeKnowledgeBase()
        result = ev._fetch_papers("test")
        assert result == []
 
    def test_total_runtime_set(self, tmp_path):
        ev  = _make_evaluator(tmp_path)
        run = ev.run(queries=["test"], methods=["tfidf"])
        assert run.total_runtime_seconds > 0
 
# -- Output writers --
 
class TestOutputWriters:
    def _make_run_with_results(self):
        from src.evaluator import EvaluationRun, QueryResult
        run = EvaluationRun()
        run.results = [
            QueryResult("Q1", "BRCA1 breast cancer", "tfidf",
                        papers_fetched=10, papers_ranked=10, top_n_evaluated=10,
                        mean_score=0.18, top1_score=0.42, score_spread=0.09,
                        graph_node_count=8, graph_edge_count=6,
                        gene_disease_pairs=5, hub_gene_count=2, hub_disease_count=1,
                        graph_density=0.3, runtime_seconds=0.04),
            QueryResult("Q1", "BRCA1 breast cancer", "biobert",
                        papers_fetched=10, papers_ranked=10, top_n_evaluated=10,
                        mean_score=0.76, top1_score=0.91, score_spread=0.11,
                        graph_node_count=9, graph_edge_count=8,
                        gene_disease_pairs=7, hub_gene_count=3, hub_disease_count=1,
                        graph_density=0.4, runtime_seconds=120.0),
        ]
        run.total_runtime_seconds = 121.0
        return run
 
    def test_save_csv_creates_file(self, tmp_path):
        from src.evaluator import save_csv
        run  = self._make_run_with_results()
        path = tmp_path / "results.csv"
        save_csv(run, path)
        assert path.exists()
 
    def test_csv_has_gcs_column(self, tmp_path):
        import csv as csv_mod
        from src.evaluator import save_csv
        run  = self._make_run_with_results()
        path = tmp_path / "results.csv"
        save_csv(run, path)
        with open(path) as f:
            reader = csv_mod.DictReader(f)
            rows   = list(reader)
        assert "gene_disease_pairs" in rows[0]
        assert len(rows) == 2
 
    def test_csv_values_correct(self, tmp_path):
        import csv as csv_mod
        from src.evaluator import save_csv
        run  = self._make_run_with_results()
        path = tmp_path / "results.csv"
        save_csv(run, path)
        with open(path) as f:
            rows = list(csv_mod.DictReader(f))
        assert rows[0]["gene_disease_pairs"] == "5"
        assert rows[1]["gene_disease_pairs"] == "7"
 
    def test_save_json_creates_file(self, tmp_path):
        from src.evaluator import save_json_summary
        run  = self._make_run_with_results()
        path = tmp_path / "summary.json"
        save_json_summary(run, path)
        assert path.exists()
 
    def test_json_has_aggregate_key(self, tmp_path):
        from src.evaluator import save_json_summary
        run  = self._make_run_with_results()
        path = tmp_path / "summary.json"
        save_json_summary(run, path)
        data = json.loads(path.read_text())
        assert "aggregate" in data
 
    def test_json_gcs_winner_biobert(self, tmp_path):
        from src.evaluator import save_json_summary
        run  = self._make_run_with_results()
        path = tmp_path / "summary.json"
        save_json_summary(run, path)
        data = json.loads(path.read_text())
        assert data["aggregate"]["gcs_winner"] == "biobert"
 
    def test_print_report_no_crash(self, tmp_path, capsys):
        from src.evaluator import print_report
        run = self._make_run_with_results()
        print_report(run)
        out = capsys.readouterr().out
        assert "BioSearch AI — Evaluation Report" in out
        assert "Mean GCS (primary)" in out
        assert "GCS:" in out
 
    def test_findings_md_created(self, tmp_path):
        from src.evaluator import generate_findings_md
        run  = self._make_run_with_results()
        path = tmp_path / "findings.md"
        generate_findings_md(run, path)
        assert path.exists()
 
    def test_findings_md_contains_outperformed(self, tmp_path):
        from src.evaluator import generate_findings_md
        run  = self._make_run_with_results()
        path = tmp_path / "findings.md"
        generate_findings_md(run, path)
        content = path.read_text()
        assert "no statistically significant difference" in content.lower()
        assert "spanning zero" in content.lower()
 
    def test_findings_md_tfidf_wins_variant(self, tmp_path):
        from src.evaluator import EvaluationRun, QueryResult, generate_findings_md
        run = EvaluationRun()
        run.results = [
            QueryResult("Q1","q","tfidf",   gene_disease_pairs=9, runtime_seconds=0.04),
            QueryResult("Q1","q","biobert", gene_disease_pairs=3, runtime_seconds=50.0),
        ]
        path = tmp_path / "findings.md"
        generate_findings_md(run, path)
        content = path.read_text()
        assert "no statistically significant difference" in content.lower()
        assert "spanning zero" in content.lower()

    def test_findings_md_contains_statistical_check(self, tmp_path):
        from src.evaluator import generate_findings_md
        run = self._make_run_with_results()
        path = tmp_path / "findings.md"
        generate_findings_md(run, path)
        content = path.read_text()
        assert "## Statistical Check" in content
        assert "sign-test" in content.lower()