"""
scripts/label_workflow.py

Relevance-label workflow for the pooled CSVs (data/relevance_Q*.csv). Every final label is set by the author.

  prepare  -> writes a blinded, shuffled input CSV (no ranks, no ranker names, no labels) for the judge file.
              ``prepare --calibration`` does this for the hand-labeled queries (default Q1-Q5) with their labels stripped.
  compare  -> scores a judge file for Q1-Q5 against the author's hand labels (read-only), records agreement
              (accuracy, Cohen's kappa) and a pass/fail gate (default kappa >= 0.6).
  import   -> validates a judge file for the other queries against the pooled CSVs (ids, completeness, 0/1/NA) and
              writes the proposed labels with notes "proposed:<judge> | conf=... | reason". Refuses unless that judge
              passed ``compare``. Existing labels are never overwritten. Papers without an abstract are left for the author.
  audit    -> ``audit make`` draws a random sample of imported rows into a BLIND sheet (proposed labels stored
              separately); ``audit score`` reports the author's agreement with them.
  accept   -> lighter route: after a scored audit (+ flagged rows reviewed), finalize the rest; see below.
  review   -> ``review make`` writes a sheet of every proposed row; the author enters the final 0/1 in ``human_label``;
              ``review apply`` writes the final labels, records how many were changed, and marks notes "reviewed: ...".

Relevance policy = the strict user-intent policy in PRECISION_RECALL_FINDINGS.md (section 4).

Typical order:
    python scripts/label_workflow.py prepare --calibration --out cal_input.csv
    python scripts/label_workflow.py compare --file cal_answer.csv
    python scripts/label_workflow.py prepare --out judge_input.csv
    python scripts/label_workflow.py import --file judge_answer.csv --dry-run
    python scripts/label_workflow.py import --file judge_answer.csv
    python scripts/label_workflow.py audit make --fraction 0.10      # label the sample, then: audit score
    python scripts/label_workflow.py review make                     # fill human_label for every row, then: review apply
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
REVIEW_DIR = DATA_DIR / "label_review"
CALIBRATION_PATH = REVIEW_DIR / "calibration.json"
AUDIT_PATH = REVIEW_DIR / "audit_sample.csv"
DEFAULT_CALIBRATION_QUERIES = ["Q1", "Q2", "Q3", "Q4", "Q5"]
DEFAULT_MIN_KAPPA = 0.6
NEEDS_HUMAN_PREFIX = "NEEDS_HUMAN"
PROPOSED_PREFIX = "proposed:"

POLICY = """\
A paper is relevant ONLY if it substantively satisfies the query's actual information need, not merely
because it shares a keyword or a broad topic with the query. Every named component of the query
(gene/protein, disease, drug, mechanism, setting) must be addressed together.
Examples of NOT relevant: a paper about a paralog of the named gene (p73 for a TP53 query); a paper on
the named disease broadly that omits an explicit named gene (Alzheimer/amyloid without APOE); a paper on
the gene without the named disease, or on a related gene instead (KRAS without lung cancer; EGFR/EREG
for a KRAS query); a generic inhibitor-resistance paper that does not specifically establish the named
target (generic TKI resistance for an EGFR-TKI query).
Judge ONLY from the title and abstract text provided. If the text is too short to establish that all
named components are addressed, answer 0."""



# --------------------------------------------------------------------------- helpers
def _qnum(qid: str) -> int:
    m = re.fullmatch(r"Q(\d+)", qid)
    if not m:
        raise ValueError(f"unexpected query id {qid!r} (expected Q<number>)")
    return int(m.group(1))


def docid(row: dict) -> str:
    """Same identity key as export_benchmark.py and the rankers: '<source>:<paper_id>'."""
    return f"{row['source'].strip()}:{row['paper_id'].strip()}"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def has_text(row: dict) -> bool:
    """Papers without an abstract cannot be judged fairly from the title alone: they always go to the human."""
    return bool(row.get("abstract_preview", "").strip())


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"query_id", "query_text", "source", "paper_id", "title", "abstract_preview", "relevant", "notes"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return fields, rows


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two binary label lists of equal length."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two non-empty lists of equal length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def confusion(human: list[int], proposed: list[int]) -> dict:
    return {
        "tp": sum(h == 1 and m == 1 for h, m in zip(human, proposed)),
        "fp": sum(h == 0 and m == 1 for h, m in zip(human, proposed)),
        "fn": sum(h == 1 and m == 0 for h, m in zip(human, proposed)),
        "tn": sum(h == 0 and m == 0 for h, m in zip(human, proposed)),
    }


def cmd_audit(args: argparse.Namespace) -> None:
    key_path = REVIEW_DIR / "audit_key.csv"
    if args.audit_cmd == "make":
        if AUDIT_PATH.exists() and not args.force:
            sys.exit(f"{_rel(AUDIT_PATH)} already exists (use --force to redraw; this discards your verdicts).")
        pool = []
        for p in sorted(DATA_DIR.glob("relevance_Q*.csv"), key=lambda p: _qnum(p.stem.replace("relevance_", ""))):
            _, rows = read_csv(p)
            pool += [(r["query_id"], docid(r), r["title"], r["abstract_preview"], r["relevant"]) for r in rows
                     if r["notes"].startswith(PROPOSED_PREFIX) and r["relevant"].strip() in ("0", "1")]
        if not pool:
            sys.exit("No imported rows found to audit (run `import` first, and audit BEFORE `review apply`).")
        k = max(1, round(len(pool) * args.fraction))
        sample = random.Random(args.seed).sample(pool, k)
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("w", newline="", encoding="utf-8") as fh:        # blind sheet: NO proposed label in here
            w = csv.writer(fh)
            w.writerow(["query_id", "docid", "query_text", "title", "abstract_preview", "human_label"])
            for q, d, t, ab, lab in sample:
                w.writerow([q, d, _query_text(q), t, ab, ""])
        with key_path.open("w", newline="", encoding="utf-8") as fh:           # key kept apart
            w = csv.writer(fh)
            w.writerow(["query_id", "docid", "proposed_label"])
            for q, d, t, ab, lab in sample:
                w.writerow([q, d, lab])
        print(f"wrote {k} of {len(pool)} imported rows -> {_rel(AUDIT_PATH)} (blind). Fill `human_label` (0/1) from the text alone; "
              f"do NOT open the data CSVs or {_rel(key_path)} until you finish.")
        return
    if not AUDIT_PATH.exists() or not key_path.exists():
        sys.exit("No audit sample/key. Run `audit make` first.")
    key = {(r["query_id"], r["docid"]): r["proposed_label"] for r in csv.DictReader(key_path.open(encoding="utf-8-sig"))}
    with AUDIT_PATH.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    done = [r for r in rows if r["human_label"].strip() in ("0", "1")]
    if len(done) < len(rows):
        print(f"WARNING: {len(rows) - len(done)} of {len(rows)} sample rows not yet labeled")
    if not done:
        sys.exit("Nothing labeled yet.")
    h = [int(r["human_label"]) for r in done]
    m = [int(key[(r["query_id"], r["docid"])]) for r in done]
    out = {"n_verified": len(done), "agreement": sum(x == y for x, y in zip(h, m)) / len(done),
           "kappa": cohen_kappa(h, m), "confusion": confusion(h, m)}
    (REVIEW_DIR / "audit_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def _query_text(qid: str) -> str:
    _, rows = read_csv(DATA_DIR / f"relevance_{qid}.csv")
    return rows[0]["query_text"].strip()


def _all_csv_paths() -> list[Path]:
    return sorted(DATA_DIR.glob("relevance_Q*.csv"), key=lambda p: _qnum(p.stem.replace("relevance_", "")))


def _read_judge_file(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path}: empty file")
    missing = {"query_id", "docid", "relevant"} - set(rows[0].keys())
    if missing:
        sys.exit(f"{path}: missing columns {sorted(missing)} (need query_id, docid, relevant[, confidence, reason])")
    for r in rows:
        r["query_id"], r["docid"] = r["query_id"].strip(), r["docid"].strip()
        r["relevant"] = r["relevant"].strip().upper()
    return rows


def cmd_prepare(args: argparse.Namespace) -> None:
    if args.queries:
        qids = args.queries
    elif args.calibration:
        qids = DEFAULT_CALIBRATION_QUERIES
    else:
        qids = [p.stem.replace("relevance_", "") for p in _all_csv_paths()
                if p.stem.replace("relevance_", "") not in DEFAULT_CALIBRATION_QUERIES
                and any(r["relevant"].strip() == "" for r in read_csv(p)[1])]
    if not qids:
        sys.exit("Nothing to prepare (no unlabeled CSVs).")
    out_rows = []
    for q in sorted(qids, key=_qnum):
        path = DATA_DIR / f"relevance_{q}.csv"
        if not path.exists():
            sys.exit(f"data/relevance_{q}.csv not found. Create it first: python scripts/run_benchmark_queries.py --only {q}")
        _, rows = read_csv(path)
        todo = rows if args.calibration else [r for r in rows if r["relevant"].strip() == ""]
        order = list(range(len(todo)))
        random.Random(f"{args.seed}:{q}").shuffle(order)
        for i in order:
            r = todo[i]
            out_rows.append({"query_id": q, "query_text": r["query_text"].strip(), "docid": docid(r),
                             "title": r["title"].strip(), "abstract": r["abstract_preview"].strip()})
    with Path(args.out).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["query_id", "query_text", "docid", "title", "abstract"])
        w.writeheader()
        w.writerows(out_rows)
    n_empty = sum(1 for r in out_rows if not r["abstract"])
    print(f"wrote {len(out_rows)} rows for {len(set(r['query_id'] for r in out_rows))} queries -> {args.out} "
          f"(blind: no ranks, ranker names or labels; {n_empty} without abstract)")


def cmd_compare(args: argparse.Namespace) -> None:
    rows = _read_judge_file(Path(args.file))
    by_q: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["docid"] in by_q.setdefault(r["query_id"], {}):
            sys.exit(f"duplicate judgment for {r['query_id']} {r['docid']}")
        by_q[r["query_id"]][r["docid"]] = r
    h_all: list[int] = []
    m_all: list[int] = []
    previews: list[int] = []
    per_query = {}
    abstained = 0
    for q in sorted(by_q, key=_qnum):
        path = DATA_DIR / f"relevance_{q}.csv"
        if not path.exists():
            sys.exit(f"data/relevance_{q}.csv not found")
        _, data = read_csv(path)
        if any(r["relevant"].strip() not in ("0", "1") for r in data):
            sys.exit(f"{path.name}: calibration needs fully hand-labeled rows (0/1)")
        index = {docid(r): r for r in data}
        unknown = set(by_q[q]) - set(index)
        if unknown:
            sys.exit(f"{q}: {len(unknown)} docids not in {path.name}, e.g. {sorted(unknown)[:3]}")
        missing = [d for d, r in index.items() if has_text(r) and d not in by_q[q]]
        if missing:
            sys.exit(f"{q}: judge file is missing {len(missing)} papers, e.g. {missing[:3]}")
        h, m = [], []
        for d, jr in by_q[q].items():
            if jr["relevant"] not in ("0", "1"):
                abstained += 1
                continue
            h.append(int(index[d]["relevant"])); m.append(int(jr["relevant"]))
            previews.append(len(index[d]["abstract_preview"]))
        h_all += h; m_all += m
        per_query[q] = {"n": len(h), "agreement": sum(a == b for a, b in zip(h, m)) / len(h)}
        print(f"  {q}: agreement {per_query[q]['agreement']:.2f} on {len(h)} papers")
    k = cohen_kappa(h_all, m_all)
    result = {"n": len(h_all), "queries": sorted(by_q, key=_qnum),
              "accuracy": sum(a == b for a, b in zip(h_all, m_all)) / len(h_all), "kappa": k,
              "confusion": confusion(h_all, m_all), "per_query": per_query, "abstained": abstained,
              "human_relevant_rate": sum(h_all) / len(h_all),
              "mean_abstract_preview_chars": sum(previews) / len(previews),
              "min_kappa_required": args.min_kappa, "passed": k >= args.min_kappa}
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cal = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else {}
    cal[args.judge] = result
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(f"{args.judge}: accuracy {result['accuracy']:.3f}, kappa {k:.3f} -> "
          f"{'PASS' if result['passed'] else 'FAIL'} (threshold {args.min_kappa}); confusion {result['confusion']}")
    if not result["passed"]:
        sys.exit("Below the kappa threshold: this judge may not be used to label.")


def cmd_import(args: argparse.Namespace) -> None:
    cal = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else {}
    c = cal.get(args.judge)
    if not c or not c.get("passed"):
        sys.exit(f"Judge {args.judge!r} has no PASSED calibration. Run `compare` on its Q1-Q5 answers first.")
    rows = _read_judge_file(Path(args.file))
    by_q: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r["docid"] in by_q.setdefault(r["query_id"], {}):
            sys.exit(f"duplicate judgment for {r['query_id']} {r['docid']}")
        by_q[r["query_id"]][r["docid"]] = r
    pending: dict[str, tuple[list[str], list[dict]]] = {}
    stats = {"labeled": 0, "abstained": 0, "kept_existing": 0, "low_confidence": 0}
    for q in sorted(by_q, key=_qnum):
        path = DATA_DIR / f"relevance_{q}.csv"
        if not path.exists():
            sys.exit(f"data/relevance_{q}.csv not found")
        fields, data = read_csv(path)
        index = {docid(r): r for r in data}
        unknown = set(by_q[q]) - set(index)
        if unknown:
            sys.exit(f"{q}: {len(unknown)} docids are not in {path.name}, e.g. {sorted(unknown)[:3]} (ids must be copied exactly)")
        missing = [d for d, r in index.items() if r["relevant"].strip() == "" and d not in by_q[q]]
        if missing:
            sys.exit(f"{q}: judge file is missing {len(missing)} unlabeled papers, e.g. {missing[:3]}")
        for d, jr in by_q[q].items():
            row = index[d]
            if row["relevant"].strip() != "":
                stats["kept_existing"] += 1          # never overwrite an existing label
                continue
            val = jr["relevant"]
            conf = (jr.get("confidence") or "NA").strip().lower()
            reason = " ".join((jr.get("reason") or "").split())[:200]
            if not has_text(row):
                if val != "NA":
                    sys.exit(f"{q} {d}: paper has no abstract, so the judge must answer NA, got {val!r}")
            elif val not in ("0", "1", "NA"):
                sys.exit(f"{q} {d}: relevant must be 0, 1 or NA, got {val!r}")
            if val in ("0", "1"):
                row["relevant"] = val
                row["notes"] = f"{PROPOSED_PREFIX}{args.judge} | conf={conf} | {reason}"
                stats["labeled"] += 1
                stats["low_confidence"] += conf == "low"
            else:
                row["notes"] = f"{NEEDS_HUMAN_PREFIX} judge_abstained | {reason}"
                stats["abstained"] += 1
        pending[q] = (fields, data)
    if args.dry_run:
        print(f"VALID (dry run, nothing written): {stats}")
        return
    for q, (fields, data) in pending.items():     # two-phase: nothing is written unless every query validated
        write_csv_atomic(DATA_DIR / f"relevance_{q}.csv", fields, data)
    print(f"imported: {stats}")
    print("Next: `audit make` (blind sample), label it by hand, `audit score`, then `review make` and check EVERY row.")


_NOTE_RE = re.compile(r"conf=(\w+) \| (.*)$")


def cmd_review(args: argparse.Namespace) -> None:
    sheet = REVIEW_DIR / "review_sheet.csv"
    if args.review_cmd == "make":
        out = []
        for p in _all_csv_paths():
            _, rows = read_csv(p)
            for r in rows:
                n = r["notes"].strip()
                if n.startswith(PROPOSED_PREFIX) or n.startswith(NEEDS_HUMAN_PREFIX):
                    m = _NOTE_RE.search(n)
                    flagged = n.startswith(NEEDS_HUMAN_PREFIX) or (m is not None and m.group(1) == "low")
                    if getattr(args, "flagged_only", False) and not flagged:
                        continue
                    out.append({"query_id": r["query_id"], "docid": docid(r), "query_text": r["query_text"],
                                "title": r["title"], "abstract_preview": r["abstract_preview"],
                                "proposed_label": r["relevant"].strip(), "proposed_confidence": m.group(1) if m else "",
                                "proposed_reason": m.group(2) if m else n, "human_label": ""})
        if not out and getattr(args, "flagged_only", False):
            print("No flagged rows (no low-confidence or abstained papers); nothing to review by hand.")
            return
        if not out:
            sys.exit("Nothing to review.")
        if sheet.exists() and not args.force:
            sys.exit(f"{_rel(sheet)} exists (use --force to redraw; this discards your entries).")
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        with sheet.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader(); w.writerows(out)
        low = sum(1 for r in out if r["proposed_confidence"] == "low")
        print(f"wrote {len(out)} rows -> {_rel(sheet)} ({low} low-confidence). Enter YOUR 0/1 in human_label for every row.")
        return
    if not sheet.exists():
        sys.exit("No review sheet. Run `review make` first.")
    with sheet.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r for r in rows if r["human_label"].strip() not in ("0", "1")]
    if bad:
        sys.exit(f"{len(bad)} of {len(rows)} rows have no valid human_label (0/1); review everything before applying.")
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["query_id"], []).append(r)
    writes = {}
    prop_l, hum_l, over = [], [], 0
    for q, srows in by_q.items():
        path = DATA_DIR / f"relevance_{q}.csv"
        fields, data = read_csv(path)
        index = {docid(r): r for r in data}
        for sr in srows:
            if sr["docid"] not in index:
                sys.exit(f"{q}: {sr['docid']} not found in {path.name}")
            row = index[sr["docid"]]
            h, l = sr["human_label"].strip(), sr["proposed_label"].strip()
            if l in ("0", "1"):
                prop_l.append(int(l)); hum_l.append(int(h)); over += (l != h)
            row["relevant"] = h
            row["notes"] = f"reviewed: proposed={l or 'none'} final={h}"
        writes[q] = (fields, data)
    for q, (fields, data) in writes.items():
        write_csv_atomic(DATA_DIR / f"relevance_{q}.csv", fields, data)
    result = {"rows_reviewed": len(rows), "rows_with_proposed_label": len(prop_l), "overridden": over,
              "override_rate": over / len(prop_l) if prop_l else None,
              "kappa_proposed_vs_final": cohen_kappa(prop_l, hum_l) if prop_l else None}
    (REVIEW_DIR / "review_result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))




def cmd_accept(args: argparse.Namespace) -> None:
    """Finalize the rows that were NOT individually reviewed, after a blind audit has been scored.

    Rows in the audited sample take the author's blind label; every other still-proposed row keeps its proposed label,
    marked "accepted:". Refuses until the audit was scored on >= --min-audit papers and no abstained rows remain open.
    """
    res_path = REVIEW_DIR / "audit_result.json"
    if not res_path.exists():
        sys.exit("No audit result. Run `audit make`, label the sample, and `audit score` first.")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    if res["n_verified"] < args.min_audit:
        sys.exit(f"Audit covers only {res['n_verified']} papers (need >= {args.min_audit}); label more of the sample and rerun `audit score`.")
    human = {}
    if AUDIT_PATH.exists():
        with AUDIT_PATH.open(newline="", encoding="utf-8-sig") as fh:
            human = {(r["query_id"], r["docid"]): r["human_label"].strip() for r in csv.DictReader(fh)
                     if r["human_label"].strip() in ("0", "1")}
    writes, open_rows, audited, overridden, accepted, rule_zero = {}, 0, 0, 0, 0, 0
    for p in _all_csv_paths():
        fields, data = read_csv(p)
        changed = False
        for r in data:
            n = r["notes"].strip()
            if n.startswith(NEEDS_HUMAN_PREFIX):
                if args.no_abstract_as_zero and not r["abstract_preview"].strip():
                    r["relevant"], r["notes"] = "0", "rule: no abstract -> not relevant"   # policy: cannot show every component is covered
                    rule_zero += 1
                    changed = True
                else:
                    open_rows += 1
            elif n.startswith(PROPOSED_PREFIX):
                l = r["relevant"].strip()
                h = human.get((r["query_id"], docid(r)))
                if h is not None:
                    audited += 1; overridden += (h != l)
                    r["relevant"], r["notes"] = h, f"reviewed: proposed={l} final={h}"
                else:
                    accepted += 1
                    r["notes"] = f"accepted: proposed={l} final={l}"
                changed = True
        if changed:
            writes[p] = (fields, data)
    if open_rows:
        sys.exit(f"{open_rows} abstained rows still need your label: run `review make --flagged-only`, fill it, `review apply`, then rerun `accept` "
                 f"(or pass --no-abstract-as-zero to label papers without an abstract 0 by rule).")
    for p, (fields, data) in writes.items():
        write_csv_atomic(p, fields, data)
    out = {"audited_rows_finalized": audited, "changed_by_audit": overridden, "accepted_unchecked": accepted, "no_abstract_labeled_0_by_rule": rule_zero,
           "audit_agreement": res["agreement"], "audit_kappa": res["kappa"], "audit_n": res["n_verified"]}
    (REVIEW_DIR / "accept_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    if res["agreement"] < 0.85:
        print("WARNING: audit agreement is below 0.85 - consider a full `review make` / `review apply` instead of accepting.")



def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Relevance-label workflow: prepare / compare / import / audit / review.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare", help="Write a blinded, shuffled input CSV for the judge file.")
    pr.add_argument("--queries", nargs="*")
    pr.add_argument("--calibration", action="store_true", help="Q1-Q5 (hand-labeled) with labels stripped, for blind calibration.")
    pr.add_argument("--out", default="judge_input.csv")
    pr.add_argument("--seed", type=int, default=13)
    cp = sub.add_parser("compare", help="Score a judge file's Q1-Q5 answers against your hand labels (read-only) and record calibration.")
    cp.add_argument("--file", required=True)
    cp.add_argument("--judge", default="judge1")
    cp.add_argument("--min-kappa", type=float, default=DEFAULT_MIN_KAPPA)
    im = sub.add_parser("import", help="Validate and import a judge file into data/relevance_Q*.csv.")
    im.add_argument("--file", required=True)
    im.add_argument("--judge", default="judge1")
    im.add_argument("--dry-run", action="store_true", help="Validate only; write nothing.")
    rv = sub.add_parser("review", help="Sheet for checking EVERY proposed label yourself.")
    rv.add_argument("review_cmd", choices=["make", "apply"])
    rv.add_argument("--force", action="store_true")
    rv.add_argument("--flagged-only", action="store_true", help="make: only low-confidence and abstained rows (use with `accept`).")
    ac = sub.add_parser("accept", help="After `audit score` (and `review apply` for flagged rows): finalize the remaining rows.")
    ac.add_argument("--min-audit", type=int, default=20)
    ac.add_argument("--no-abstract-as-zero", action="store_true",
                    help="label papers with no abstract 0 by rule (policy: components cannot be shown) instead of reviewing them by hand")
    a = sub.add_parser("audit", help="Blind sample of imported rows for you to label yourself.")
    a.add_argument("audit_cmd", choices=["make", "score"])
    a.add_argument("--fraction", type=float, default=0.10)
    a.add_argument("--seed", type=int, default=13)
    a.add_argument("--force", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    {"audit": cmd_audit, "prepare": cmd_prepare, "compare": cmd_compare,
     "import": cmd_import, "review": cmd_review, "accept": cmd_accept}[args.cmd](args)


if __name__ == "__main__":
    main()
