"""
scripts/run_benchmark_queries.py

Automates the NON-manual part of growing the benchmark: for every query in benchmark/queries.json that
has no data/relevance_<qid>.csv yet, call your existing build_relevance_annotation_template.py (fetch ->
dedup -> TF-IDF + BioBERT top-10 -> pooled CSV with an empty `relevant` column).

It never writes labels. Afterwards YOU open each new CSV and fill `relevant` with 0/1 under the strict
user-intent policy in PRECISION_RECALL_FINDINGS.md, then run:

    python scripts/export_benchmark.py
    python benchmark/evaluate.py --qrels benchmark/qrels.tsv \\
        --run tfidf=benchmark/runs/tfidf.run --run biobert=benchmark/runs/biobert.run --out-dir benchmark

Requires NCBI_EMAIL in .env, network access and the BioBERT weights (same as the template script).
    python scripts/run_benchmark_queries.py [--only Q6 Q7] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def unlabeled_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for r in csv.DictReader(fh) if r.get("relevant", "").strip() == "")


def check_pool(path: Path, top_n: int = 10) -> None:
    """Fail loudly if the pooled CSV lacks a full top-N for BOTH rankers (e.g. BioBERT silently unavailable)."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for col in ("tfidf_rank", "biobert_rank"):
        got = sorted(int(r[col]) for r in rows if r[col].strip() and int(r[col]) <= top_n)
        if got != list(range(1, top_n + 1)):
            sys.exit(f"{path.name}: {col} does not contain ranks 1..{top_n} exactly once ({got}). "
                     f"Too few candidates fetched, or BioBERT failed to load - fix or replace this query before labeling.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "benchmark" / "queries.json"))
    ap.add_argument("--only", nargs="*", help="restrict to these query ids")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    queries = json.loads(Path(a.queries).read_text(encoding="utf-8"))["queries"]
    unknown = set(a.only or []) - {q["qid"] for q in queries}
    if unknown:
        sys.exit(f"unknown query ids in --only: {sorted(unknown)}")
    todo = 0
    for q in queries:
        if a.only and q["qid"] not in a.only:
            continue
        out = ROOT / "data" / f"relevance_{q['qid']}.csv"
        if out.exists():
            n = unlabeled_count(out)
            print(f"{q['qid']}: exists ({n} unlabeled rows)" if n else f"{q['qid']}: exists, fully labeled")
            todo += n
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "build_relevance_annotation_template.py"),
               "--query", q["query"], "--query-id", q["qid"], "--out", str(out),
               "--abstract-chars", "1500"]  # longer previews give the judges (and you) real evidence
        print(f"{q['qid']}: {' '.join(cmd[1:])}")
        if not a.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)
            check_pool(out)
            todo += unlabeled_count(out)
    print(f"\nLabels still to be filled in by hand: {todo}")


if __name__ == "__main__":
    main()
