"""Fast, model-free tests for the benchmark bundle. Included in the CI 'not slow and not integration' run.

The key test is a REPRODUCTION CHECK: starting only from the committed hand-labeled CSVs, the exported qrels/runs
and the benchmark evaluator must reproduce the per-query P@10 stored in data/precision_recall_summary.json.
"""
import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from benchmark.evaluate import read_qrels, read_run, score_query

ROOT = Path(__file__).resolve().parent.parent


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_benchmark", ROOT / "scripts" / "export_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reproduces_committed_precision_at_10(tmp_path):
    exp = _load_exporter()
    summary = json.loads((ROOT / "data" / "precision_recall_summary.json").read_text())
    # Only the queries that are in the committed summary (Q1-Q5): Q6+ may be mid-labeling and must not break CI.
    ids = {row["query_id"] for row in summary["per_query"]}
    paths = sorted(str(ROOT / "data" / f"relevance_{q}.csv") for q in ids)
    assert paths, "no hand-labeled CSVs found"
    exp.export(paths, tmp_path)
    qrels = read_qrels(str(tmp_path / "qrels.tsv"))
    runs = {m: read_run(str(tmp_path / "runs" / f"{m}.run")) for m in ("tfidf", "biobert")}
    checked = 0
    for row in summary["per_query"]:
        qid = row["query_id"]
        for m in ("tfidf", "biobert"):
            got, _ = score_query(runs[m][qid], qrels[qid])
            assert math.isclose(got["P@10"], row[m]["precision_at_10"], abs_tol=1e-9), (qid, m)
            checked += 1
    assert checked >= 10


def test_exporter_refuses_unlabeled_rows(tmp_path):
    exp = _load_exporter()
    src = ROOT / "data" / "relevance_Q1.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    rows[0]["relevant"] = ""
    bad = tmp_path / "relevance_Q1.csv"
    with bad.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="unlabeled"):
        exp.export([str(bad)], tmp_path / "out")


def test_exporter_refuses_incomplete_ranks(tmp_path):
    exp = _load_exporter()
    rows = list(csv.DictReader((ROOT / "data" / "relevance_Q1.csv").open(encoding="utf-8")))
    for r in rows:
        if r["tfidf_rank"] == "10":
            r["tfidf_rank"] = ""
    bad = tmp_path / "relevance_Q1.csv"
    with bad.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="each rank 1..10"):
        exp.export([str(bad)], tmp_path / "out")


def test_queries_file_is_well_formed():
    q = json.loads((ROOT / "benchmark" / "queries.json").read_text())["queries"]
    assert len(q) == 25
    assert len({x["qid"] for x in q}) == 25 and len({x["query"] for x in q}) == 25
    # Q1-Q5 must stay identical to the repo's standard evaluation queries
    from src.evaluator import EVALUATION_QUERIES
    assert [x["query"] for x in q[:5]] == EVALUATION_QUERIES


def test_score_query_edge_cases():
    zero, _ = score_query(["a", "b"], {"a": 0, "b": 0})
    assert zero == {"P@10": 0.0, "MRR@10": 0.0, "nDCG@10": 0.0}
    empty, _ = score_query([], {"a": 1})
    assert empty["P@10"] == 0.0 and empty["nDCG@10"] == 0.0
    perfect, unj = score_query(["a", "b"], {"a": 1, "b": 1})
    assert perfect["nDCG@10"] == 1.0 and perfect["MRR@10"] == 1.0 and math.isclose(perfect["P@10"], 0.2)
    _, unj = score_query(["a", "x"], {"a": 1})
    assert math.isclose(unj, 0.1)


def test_bootstrap_and_sign_test_edge_cases():
    from benchmark.evaluate import boot_ci, sign_test_p, qid_key
    lo, hi = boot_ci([])
    assert math.isnan(lo) and math.isnan(hi)
    assert boot_ci([0.5]) == (0.5, 0.5)
    assert boot_ci([0.2, 0.4, 0.6]) == boot_ci([0.2, 0.4, 0.6])  # fixed seed -> deterministic
    assert sign_test_p([0, 0, 0]) is None
    assert math.isclose(sign_test_p([1, 1, 1, 1, 1]), 0.0625)
    assert sorted(["Q10", "Q2", "Q1"], key=qid_key) == ["Q1", "Q2", "Q10"]


def test_evaluator_cli_runs_end_to_end_on_tied_systems(tmp_path):
    import subprocess, sys
    exp = _load_exporter()
    summary = json.loads((ROOT / "data" / "precision_recall_summary.json").read_text())
    paths = sorted(str(ROOT / "data" / f"relevance_{r['query_id']}.csv") for r in summary["per_query"])
    exp.export(paths, tmp_path)
    run = tmp_path / "runs" / "tfidf.run"
    cmd = [sys.executable, str(ROOT / "benchmark" / "evaluate.py"), "--qrels", str(tmp_path / "qrels.tsv"),
           "--run", f"a={run}", "--run", f"b={run}", "--out-dir", str(tmp_path / "res")]
    subprocess.run(cmd, check=True, cwd=ROOT)  # identical systems: every difference is zero, must not crash
    assert "NA(all-ties)" in (tmp_path / "res" / "eval_summary.csv").read_text()


def test_manifest_hashes_files(tmp_path):
    spec = importlib.util.spec_from_file_location("make_manifest", ROOT / "scripts" / "make_manifest.py")
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    (tmp_path / "x.txt").write_text("hi")
    m = mm.build_manifest(tmp_path)
    assert m["x.txt"]["bytes"] == 2 and len(m["x.txt"]["sha256"]) == 64


def test_queries_are_topically_diverse():
    q = json.loads((ROOT / "benchmark" / "queries.json").read_text())["queries"]
    assert sum("onco" in x["category"] for x in q) <= 6, "benchmark should not be dominated by cancer queries"
    assert len({x["category"] for x in q}) >= 15


def test_merge_judge_answers_strips_fences_and_rejects_bad_rows(tmp_path):
    spec = importlib.util.spec_from_file_location("merge_judge_answers", ROOT / "scripts" / "merge_judge_answers.py")
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    good = tmp_path / "r1.txt"
    good.write_text('```csv\nquery_id,docid,relevant,confidence,reason\nQ6,pubmed:1,1,high,"covers all parts, yes"\nQ6,pubmed:2,na,low,"no abstract"\n```\nROWS: 2 in / 2 out\n')
    rows = mj.merge([str(good)])
    assert rows[0][4] == "covers all parts, yes" and rows[1][2] == "NA" and len(rows) == 2
    bad = tmp_path / "r2.txt"
    bad.write_text("Q6,pubmed:3,2,high,x\n")
    with pytest.raises(ValueError, match="0, 1 or NA"):
        mj.merge([str(bad)])
    dup = tmp_path / "r3.txt"
    dup.write_text("Q6,pubmed:1,1,high,x\n")
    with pytest.raises(ValueError, match="duplicate"):
        mj.merge([str(good), str(dup)])
