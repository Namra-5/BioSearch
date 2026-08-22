"""
scripts/compute_precision_recall.py

Compute precision@N and pooled recall@N from hand-labeled rankings.

Reads a CSV produced by build_relevance_annotation_template.py AFTER you
have hand-filled the `relevant` column with 0/1, and computes:

  - precision@10 per method  = (# relevant papers in that method's top-10)
                                / 10
  - pooled recall@10 per method = (# relevant papers in that method's top-10)
                                             / (# relevant papers in the union of both
                                                 methods' top-10)

METHODOLOGY NOTE
----------------------------------------------------------------------------
Recall here is "pooled recall," the standard TREC-style approximation used
whenever exhaustively judging an entire corpus is infeasible: the
denominator is the number of relevant papers found within the UNION of
both methods' top-10, not the true corpus-wide number of relevant papers.
The output names this metric pooled recall to distinguish it from exhaustive
corpus recall.

USAGE
-------
    python scripts/compute_precision_recall.py --in data/relevance_Q1.csv

    # Multiple queries at once (glob):
    python scripts/compute_precision_recall.py --in data/relevance_Q*.csv

Prints a per-query table and an aggregate summary. Also writes a
`data/precision_recall_summary.json` you can pull real numbers from
straight into findings.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def _load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _validate_rows(rows: list[dict], path: str, top_n: int) -> None:
    unlabeled = [r for r in rows if r.get("relevant", "").strip() == ""]
    if unlabeled:
        raise ValueError(
            f"{path}: {len(unlabeled)} of {len(rows)} rows still have an "
            f"empty 'relevant' column. Label every row (0 or 1) before "
            f"computing precision/recall — a partially-labeled file will "
            f"silently produce wrong numbers if we just skip blanks, so "
            f"this script refuses to guess for you."
        )
    bad = [r for r in rows if r["relevant"].strip() not in ("0", "1")]
    if bad:
        raise ValueError(
            f"{path}: found {len(bad)} rows where 'relevant' is not "
            f"exactly '0' or '1'. Fix these before re-running."
        )

    for method, rank_col in (("tfidf", "tfidf_rank"), ("biobert", "biobert_rank")):
        ranked = [r for r in rows if r.get(rank_col, "").strip()]
        invalid = []
        for row in ranked:
            try:
                rank = int(row[rank_col])
            except (TypeError, ValueError):
                invalid.append(row[rank_col])
                continue
            if rank < 1:
                invalid.append(row[rank_col])
        if invalid:
            raise ValueError(
                f"{path}: {method} contains invalid ranks {invalid!r}; "
                "ranks must be positive integers."
            )

        top_rows = [r for r in ranked if int(r[rank_col]) <= top_n]
        ranks = [int(r[rank_col]) for r in top_rows]
        expected_ranks = set(range(1, top_n + 1))
        if set(ranks) != expected_ranks or len(ranks) != top_n:
            raise ValueError(
                f"{path}: expected exactly one {method} candidate at every "
                f"rank 1 through {top_n}; found ranks {sorted(ranks)!r}."
            )

        paper_keys = [(r.get("source", ""), r.get("paper_id", "")) for r in top_rows]
        if len(set(paper_keys)) != len(paper_keys):
            raise ValueError(
                f"{path}: {method} top-{top_n} contains duplicate paper "
                "identities."
            )


def _compute_for_query(rows: list[dict], top_n: int = 10) -> dict:
    query_id = rows[0]["query_id"]
    # The pooled set is explicitly the union of papers ranked in either
    # method's top-N. Unranked labeled rows do not enter the recall denominator.
    pooled_paper_keys = {
        (r["source"], r["paper_id"])
        for r in rows
        if (
            (r["tfidf_rank"].strip() and int(r["tfidf_rank"]) <= top_n)
            or
            (r["biobert_rank"].strip() and int(r["biobert_rank"]) <= top_n)
        )
    }
    relevant_paper_keys = {
        (r["source"], r["paper_id"])
        for r in rows
        if (r["source"], r["paper_id"]) in pooled_paper_keys
        and r["relevant"] == "1"
    }
    total_relevant_pooled = len(relevant_paper_keys)

    result = {"query_id": query_id, "pooled_relevant_count": total_relevant_pooled}

    for method, rank_col in (("tfidf", "tfidf_rank"), ("biobert", "biobert_rank")):
        in_top_n = [
            r for r in rows
            if r[rank_col].strip() != "" and int(r[rank_col]) <= top_n
        ]
        relevant_in_top_n = sum(1 for r in in_top_n if r["relevant"] == "1")
        n_ranked = len(in_top_n)

        # Precision@N is defined over the requested cutoff, including any
        # missing results as non-relevant rather than changing the denominator.
        precision = relevant_in_top_n / top_n
        recall = (
            relevant_in_top_n / total_relevant_pooled
            if total_relevant_pooled else 0.0
        )
        result[method] = {
            "n_in_top_n": n_ranked,
            "relevant_in_top_n": relevant_in_top_n,
            f"precision_at_{top_n}": round(precision, 4),
            f"pooled_recall_at_{top_n}": round(recall, 4),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute precision@10 / pooled recall@10.")
    parser.add_argument("--in", dest="input_glob", required=True,
                         help="CSV path or glob, e.g. data/relevance_Q1.csv or data/relevance_Q*.csv")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out", default="data/precision_recall_summary.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    all_results = []
    for path in paths:
        rows = _load_rows(path)
        if not rows:
            print(f"Skipping {path}: empty file.")
            continue
        _validate_rows(rows, path, top_n=args.top_n)
        result = _compute_for_query(rows, top_n=args.top_n)
        all_results.append(result)

    # ── Print a readable table ──────────────────────────────────────────
    print(f"\n{'Query':<8}{'Method':<10}{'P@' + str(args.top_n):<10}"
          f"{'PooledR@' + str(args.top_n):<14}{'#Rel(pooled)':<14}")
    print("-" * 56)
    for r in all_results:
        for method in ("tfidf", "biobert"):
            m = r[method]
            print(
                f"{r['query_id']:<8}{method:<10}"
                f"{m[f'precision_at_{args.top_n}']:<10}"
                f"{m[f'pooled_recall_at_{args.top_n}']:<14}"
                f"{r['pooled_relevant_count']:<14}"
            )

    # ── Aggregate mean precision/recall per method across queries ──────
    def _mean(key_path):
        method, field = key_path
        vals = [r[method][field] for r in all_results]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "per_query": all_results,
        "aggregate": {
            "tfidf": {
                f"mean_precision_at_{args.top_n}": _mean(("tfidf", f"precision_at_{args.top_n}")),
                f"mean_pooled_recall_at_{args.top_n}": _mean(("tfidf", f"pooled_recall_at_{args.top_n}")),
            },
            "biobert": {
                f"mean_precision_at_{args.top_n}": _mean(("biobert", f"precision_at_{args.top_n}")),
                f"mean_pooled_recall_at_{args.top_n}": _mean(("biobert", f"pooled_recall_at_{args.top_n}")),
            },
        },
        "n_queries_labeled": len(all_results),
        "methodology_note": (
            "Recall is POOLED recall (TREC-style): denominator is the number "
            "of relevant papers found in the union of both methods' top-N, "
            "not the true corpus-wide relevant count, which is infeasible to "
            "determine exhaustively. Relevance judgments are human-labeled "
            "by the project author, not automatically generated."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary to {out_path}")
    print(f"n_queries_labeled = {len(all_results)} "
          f"(cite this honestly — do not imply full 5-query coverage if fewer were labeled)")


if __name__ == "__main__":
    main()
