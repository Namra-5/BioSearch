"""
scripts/export_benchmark.py

Convert the hand-labeled pooled CSVs (data/relevance_Q*.csv, produced by
build_relevance_annotation_template.py and filled in by you) into a standard,
tool-independent benchmark bundle under benchmark/:

    benchmark/qrels.tsv          TREC qrels   (qid 0 docid grade)   grade in {0,1}
    benchmark/runs/tfidf.run     TREC run     (qid Q0 docid rank score runname)
    benchmark/runs/biobert.run
    benchmark/candidates.csv     id/title/url of every judged document (NO abstracts)
    benchmark/label_provenance.json   how many labels come from each source (author / author-reviewed)

and a BEIR-layout copy (same judgments, standard BEIR file names) under benchmark/beir/:

    benchmark/beir/corpus.jsonl      {"_id","title","text","metadata"}   text = abstract per --abstracts
    benchmark/beir/queries.jsonl     {"_id","text"}
    benchmark/beir/qrels/test.tsv    header "query-id<TAB>corpus-id<TAB>score", then one judgment per line

Query ids keep the repo's "Q1".."Q25" form (BEIR ids are free-form strings), so they match the CSVs and run files.

docid = "<source>:<paper_id>"  (same identity key the rankers use, e.g. pubmed:17922257)

It applies the same strictness as compute_precision_recall.py: every row must be labeled 0/1,
and each method must have exactly one paper at every rank 1..10 with no duplicates.
Run scores are synthesized from the stored ranks (score = 1/rank) because the CSVs store ranks, not
raw similarity scores; only the ORDER matters for P@10 / MRR@10 / nDCG@10.

    python scripts/export_benchmark.py [--glob "data/relevance_Q*.csv"] [--out benchmark]
                                       [--abstracts none|preview|full] [--db data/biosearch.db]

--abstracts: none (default) = title only, corpus text is ""; preview = the abstract_preview stored in the CSVs
(truncated); full = complete abstract from the local paper cache (needs --db; publisher copyright may apply).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOP_N = 10


def docid(row: dict) -> str:
    return f"{row['source'].strip()}:{row['paper_id'].strip()}"


def load_query_file(path: str) -> tuple[str, list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: empty file")
    qid = rows[0]["query_id"].strip()
    if any(r["query_id"].strip() != qid for r in rows):
        raise ValueError(f"{path}: mixed query_id values")
    blank = [r for r in rows if r.get("relevant", "").strip() == ""]
    if blank:
        raise ValueError(f"{path}: {len(blank)} of {len(rows)} rows still unlabeled - finish labeling first")
    bad = [r for r in rows if r["relevant"].strip() not in ("0", "1")]
    if bad:
        raise ValueError(f"{path}: 'relevant' must be exactly 0 or 1")
    seen = set()
    for r in rows:
        k = docid(r)
        if k in seen:
            raise ValueError(f"{path}: duplicate document {k}")
        seen.add(k)
    for col in ("tfidf_rank", "biobert_rank"):
        ranks = sorted(int(r[col]) for r in rows if r[col].strip() and int(r[col]) <= TOP_N)
        if ranks != list(range(1, TOP_N + 1)):
            raise ValueError(f"{path}: {col} must contain each rank 1..{TOP_N} exactly once, found {ranks}")
    return qid, rows


def qnum(qid: str) -> int:
    m = re.fullmatch(r"Q(\d+)", qid)
    if not m:
        raise ValueError(f"unexpected query id {qid!r} (expected Q<number>)")
    return int(m.group(1))


def export(paths: list[str], out_dir: Path, abstracts: str = "none", db_path: str | None = None) -> dict:
    loaded = sorted((load_query_file(p) for p in paths), key=lambda t: qnum(t[0]))
    qids = [q for q, _ in loaded]
    if len(set(qids)) != len(qids):
        raise ValueError("the same query_id appears in more than one file")
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    qrels, cands = [], []
    runs = {"tfidf": [], "biobert": []}
    for qid, rows in loaded:
        for r in rows:
            d = docid(r)
            qrels.append(f"{qid}\t0\t{d}\t{r['relevant'].strip()}")
            cands.append({"qid": qid, "docid": d, "source": r["source"], "paper_id": r["paper_id"],
                          "title": r["title"], "url": r.get("url", "")})
        for method, col in (("tfidf", "tfidf_rank"), ("biobert", "biobert_rank")):
            top = sorted((int(r[col]), docid(r)) for r in rows if r[col].strip() and int(r[col]) <= TOP_N)
            for rank, d in top:
                runs[method].append(f"{qid} Q0 {d} {rank} {1.0 / rank:.6f} {method}")
    (out_dir / "qrels.tsv").write_text("\n".join(qrels) + "\n", encoding="utf-8")
    for m, lines in runs.items():
        (out_dir / "runs" / f"{m}.run").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "docid", "source", "paper_id", "title", "url"])
        w.writeheader()
        w.writerows(cands)
    prov: dict[str, int] = {}
    for _, rows in loaded:
        for r in rows:
            k = provenance_of(r)
            prov[k] = prov.get(k, 0) + 1
    (out_dir / "label_provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    beir_counts = write_beir(loaded, out_dir, abstracts, db_path)
    return {"queries": qids, "judgments": len(qrels), "provenance": prov, "beir": beir_counts}


def provenance_of(row: dict) -> str:
    note = row.get("notes", "").strip()
    if note.startswith("reviewed:"):
        return "author_reviewed"
    if note.startswith("rule:"):
        return "no_abstract_rule"
    if note.startswith("accepted:"):
        return "proposed_audited_accepted"
    if note.startswith("proposed:"):
        return "proposed_unreviewed"
    if note.startswith("NEEDS_HUMAN"):
        return "author_labeled_after_abstain"
    return "human"


def _abstract_for(row: dict, mode: str, storage) -> str:
    if mode == "none":
        return ""
    if mode == "preview":
        return row.get("abstract_preview", "").strip()
    paper = storage.get_paper(row["paper_id"].strip(), row["source"].strip())
    if paper is None or not (paper.abstract or "").strip():
        raise ValueError(f"full abstract not found in cache for {docid(row)}; rerun the fetch or use --abstracts preview")
    return paper.abstract.strip()


def write_beir(loaded: list[tuple[str, list[dict]]], out_dir: Path, abstracts: str = "none", db_path: str | None = None) -> dict:
    """Write benchmark/beir/{corpus.jsonl,queries.jsonl,qrels/test.tsv}. Returns counts."""
    storage = None
    if abstracts == "full":
        from config import DB_PATH
        from src.storage import PaperStorage
        storage = PaperStorage(db_path=Path(db_path) if db_path else DB_PATH)
    beir = out_dir / "beir"
    (beir / "qrels").mkdir(parents=True, exist_ok=True)
    corpus: dict[str, dict] = {}
    queries, qrels = [], []
    for qid, rows in loaded:
        queries.append({"_id": qid, "text": rows[0]["query_text"].strip()})
        for r in rows:
            d = docid(r)
            qrels.append(f"{qid}\t{d}\t{int(r['relevant'].strip())}")
            if d not in corpus:
                corpus[d] = {"_id": d, "title": r["title"].strip(), "text": _abstract_for(r, abstracts, storage),
                             "metadata": {"source": r["source"].strip(), "paper_id": r["paper_id"].strip(),
                                          "url": r.get("url", "").strip()}}
    with (beir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for doc in corpus.values():
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with (beir / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    (beir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "\n".join(qrels) + "\n", encoding="utf-8")
    return {"corpus_docs": len(corpus), "queries": len(queries), "qrels": len(qrels)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="data/relevance_Q*.csv")
    ap.add_argument("--out", default="benchmark")
    ap.add_argument("--abstracts", choices=["none", "preview", "full"], default="none",
                    help="what goes into beir/corpus.jsonl `text` (default none = titles only)")
    ap.add_argument("--db", help="paper cache DB for --abstracts full (default: config.DB_PATH)")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="export even if some labels have not been through `review apply` (not for a release)")
    a = ap.parse_args()
    paths = sorted(glob.glob(a.glob))
    if not paths:
        sys.exit(f"no files match {a.glob}")
    try:
        info = export(paths, Path(a.out), a.abstracts, a.db)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if info["provenance"].get("proposed_unreviewed") and not a.allow_unreviewed:
        sys.exit(f"ERROR: {info['provenance']['proposed_unreviewed']} labels are still unreviewed - finish `review make` / "
                 f"`review apply` first (or pass --allow-unreviewed for a draft export). Output was written but must not be released.")
    print(f"exported {len(info['queries'])} queries / {info['judgments']} judgments -> {a.out}/ (+ {a.out}/beir/)")
    print(f"label provenance: {info['provenance']}")


if __name__ == "__main__":
    main()
