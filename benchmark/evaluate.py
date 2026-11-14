#!/usr/bin/env python3
"""Step 3 - evaluate runs with qrels (TREC format).

Metrics @ k=10: Precision@10, MRR@10 (reciprocal rank of the first relevant doc within the top 10; 0 if none), nDCG@10 (linear gain = grade; binary qrels give binary gain).
Unjudged documents count as non-relevant (standard TREC convention; report the unjudged@10 rate).
Significance: paired two-sided Wilcoxon signed-rank on per-query scores (zero differences dropped) AND an exact two-sided sign test
(binomial, ties dropped) as a distribution-free cross-check; with many tied scores the Wilcoxon p-value is a normal approximation.
Uncertainty: percentile bootstrap over queries, 1,000 resamples, seed fixed, 95% CI for each system mean and for the paired difference.
Outputs: eval_results.csv (per-query), eval_summary.csv (aggregates).
"""
import argparse, csv, math, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest, wilcoxon

K = 10
METRICS = ["P@10", "MRR@10", "nDCG@10"]


def read_qrels(path):
    q = defaultdict(dict)
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        p = line.split()
        if not p:
            continue
        if len(p) != 4:
            sys.exit(f"{path}:{ln}: qrels need 4 fields (qid 0 docid grade)")
        q[p[0]][p[2]] = int(p[3])
    return q


def read_run(path):
    run = defaultdict(list)
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        p = line.split()
        if not p:
            continue
        if len(p) != 6:
            sys.exit(f"{path}:{ln}: run needs 6 fields (qid Q0 docid rank score runname)")
        run[p[0]].append((int(p[3]), p[2]))
    out = {}
    for q, v in run.items():
        docs = [d for _, d in v]
        if len(set(docs)) != len(docs):
            sys.exit(f"{path}: query {q} ranks the same document more than once")
        out[q] = [d for _, d in sorted(v)]
    return out


def dcg(grades):
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def score_query(ranking, rels):
    top = ranking[:K]
    g = [rels.get(d, 0) for d in top]
    p10 = sum(1 for x in g if x >= 1) / K
    rr = 0.0
    for i, d in enumerate(top):  # MRR@10: judgments exist only for pooled top-10 documents
        if rels.get(d, 0) >= 1:
            rr = 1.0 / (i + 1)
            break
    ideal = sorted(rels.values(), reverse=True)[:K]
    idcg = dcg(ideal)
    ndcg = dcg(g) / idcg if idcg > 0 else 0.0
    unjudged = sum(1 for d in top if d not in rels) / K
    return {"P@10": p10, "MRR@10": rr, "nDCG@10": ndcg}, unjudged


def qid_key(q):
    m = re.fullmatch(r"([A-Za-z]*)(\d+)", q)
    return (m.group(1), int(m.group(2))) if m else (q, 0)


def sign_test_p(d):
    """Exact two-sided sign test on paired differences; ties dropped. Returns None if no non-tied pair."""
    pos, neg = int((np.asarray(d) > 0).sum()), int((np.asarray(d) < 0).sum())
    return None if pos + neg == 0 else float(binomtest(pos, pos + neg, 0.5).pvalue)


def boot_ci(x, n_boot=1000, seed=2026):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if x.size == 0:
        return float("nan"), float("nan")
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default="qrels.tsv")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH", help="repeat for each system (exactly 2 for the Wilcoxon test)")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    import os
    os.makedirs(a.out_dir, exist_ok=True)
    qrels = read_qrels(a.qrels)
    systems = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        systems[name] = read_run(path)
    qids = sorted(qrels, key=qid_key)
    for name, run in systems.items():
        missing = [q for q in qids if q not in run]
        if missing:
            sys.exit(f"system {name} has no ranking for {len(missing)} judged queries, e.g. {missing[:3]}")
    per = {n: {m: [] for m in METRICS} for n in systems}
    unj = {n: [] for n in systems}
    with open(f"{a.out_dir}/eval_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "system", "P@10", "MRR@10", "nDCG@10", "unjudged@10"])
        for q in qids:
            for n, run in systems.items():
                s, u = score_query(run[q], qrels[q])
                for m in METRICS:
                    per[n][m].append(s[m])
                unj[n].append(u)
                w.writerow([q, n, f"{s['P@10']:.4f}", f"{s['MRR@10']:.4f}", f"{s['nDCG@10']:.4f}", f"{u:.2f}"])
    rows = []
    names = list(systems)
    for n in names:
        for m in METRICS:
            lo, hi = boot_ci(per[n][m])
            rows.append([n, m, f"{np.mean(per[n][m]):.4f}", f"{lo:.4f}", f"{hi:.4f}", "", "", ""])
    if len(names) == 2:
        a_, b_ = names
        for m in METRICS:
            d = np.array(per[a_][m]) - np.array(per[b_][m])
            lo, hi = boot_ci(d)
            if np.all(d == 0):  # Wilcoxon is undefined when every paired difference is zero
                p = "NA(all-ties)"
            else:
                # scipy default: zero differences dropped; exact p only when n<=50 AND no ties/zeros, else normal approx.
                p = f"{wilcoxon(per[a_][m], per[b_][m]).pvalue:.4g}"
            sp = sign_test_p(d)
            rows.append([f"{a_} - {b_}", m, f"{d.mean():.4f}", f"{lo:.4f}", f"{hi:.4f}", p,
                         f"{int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}",
                         "NA(all-ties)" if sp is None else f"{sp:.4g}"])
    for n in names:
        rows.append([n, "unjudged@10 (mean)", f"{np.mean(unj[n]):.4f}", "", "", "", "", ""])
    with open(f"{a.out_dir}/eval_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system_or_contrast", "metric", "mean", "ci95_lo", "ci95_hi", "wilcoxon_p", "wins/losses/ties", "sign_test_p"])
        w.writerows(rows)
    print(f"{len(qids)} queries; wrote eval_results.csv and eval_summary.csv to {a.out_dir}")


if __name__ == "__main__":
    main()
