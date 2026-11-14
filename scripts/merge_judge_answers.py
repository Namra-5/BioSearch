"""scripts/merge_judge_answers.py - merge judge replies (one text file per reply) into ONE clean answer CSV.

Save each reply into its own text file (the ```csv fences and the final "ROWS: ..." line may stay in;
they are stripped here). Then:

    python scripts/merge_judge_answers.py reply1.txt reply2.txt reply3.txt --out judge_answer.csv

Checks: exact header, 5 columns per row, relevant in {0,1,NA}, no docid repeated within a query. It does NOT touch data/*.csv;
labels are written only by `label_workflow.py import`, which validates every id against the pooled CSVs.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys

HEADER = ["query_id", "docid", "relevant", "confidence", "reason"]


def parse_reply(text: str) -> list[list[str]]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("```") and not ln.strip().upper().startswith("ROWS:")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if rows and [c.strip().lower() for c in rows[0]] == HEADER:
        rows = rows[1:]
    return rows


def merge(paths: list[str]) -> list[list[str]]:
    out, seen = [], set()
    for p in paths:
        with open(p, encoding="utf-8-sig") as fh:
            rows = parse_reply(fh.read())
        for i, r in enumerate(rows, 1):
            if len(r) != 5:
                raise ValueError(f"{p}: data row {i} has {len(r)} columns, expected 5: {r[:3]}...")
            r = [c.strip() for c in r]
            r[2] = r[2].upper()
            if r[2] not in ("0", "1", "NA"):
                raise ValueError(f"{p}: row {i} ({r[0]} {r[1]}): relevant must be 0, 1 or NA, got {r[2]!r}")
            key = (r[0], r[1])
            if key in seen:
                raise ValueError(f"{p}: duplicate answer for {r[0]} {r[1]}")
            seen.add(key)
            out.append(r)
    if not out:
        raise ValueError("no answer rows found")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="judge_answer.csv")
    a = ap.parse_args()
    try:
        rows = merge(a.files)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"merged {len(rows)} answers from {len(a.files)} file(s) -> {a.out}")


if __name__ == "__main__":
    main()
