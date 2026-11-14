"""Model-free tests for scripts/label_workflow.py and the BEIR/provenance export.

The judge file is simulated from a title->label map, so these run offline with no network or model weights.
"""
import csv
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


al = _load("label_workflow", "scripts/label_workflow.py")
ex = _load("export_benchmark_mod", "scripts/export_benchmark.py")




@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for q in ("Q1", "Q2"):
        shutil.copy(ROOT / "data" / f"relevance_{q}.csv", data / f"relevance_{q}.csv")
    monkeypatch.setattr(al, "DATA_DIR", data)
    monkeypatch.setattr(al, "REVIEW_DIR", data / "label_review")
    monkeypatch.setattr(al, "CALIBRATION_PATH", data / "label_review" / "calibration.json")
    monkeypatch.setattr(al, "AUDIT_PATH", data / "label_review" / "audit_sample.csv")
    _, rows = al.read_csv(data / "relevance_Q1.csv")
    _, rows2 = al.read_csv(data / "relevance_Q2.csv")
    t2l = {r["title"]: int(r["relevant"]) for r in rows + rows2}
    return data, t2l


def _args(*argv):
    return al._build_parser().parse_args(list(argv))


def _unlabeled_copy(data, src="Q2", dst="Q6"):
    fields, rows = al.read_csv(data / f"relevance_{src}.csv")
    for r in rows:
        r["query_id"], r["relevant"], r["notes"] = dst, "", ""
    al.write_csv_atomic(data / f"relevance_{dst}.csv", fields, rows)
    return fields, rows


def test_kappa_known_values():
    assert al.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert al.cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert al.cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0          # degenerate: no chance disagreement possible


















def test_export_beir_layout_and_provenance(tmp_path):
    # Only the original hand-labeled Q1-Q5: Q6+ pools may exist unlabeled while labeling is in progress.
    paths = [str(ROOT / "data" / f"relevance_Q{i}.csv") for i in range(1, 6)]
    info = ex.export(paths, tmp_path, abstracts="preview")
    beir = tmp_path / "beir"
    assert info["provenance"] == {"human": 67}
    lines = (beir / "qrels" / "test.tsv").read_text().splitlines()
    assert lines[0] == "query-id\tcorpus-id\tscore" and len(lines) == 68
    corpus = [json.loads(l) for l in (beir / "corpus.jsonl").read_text().splitlines()]
    queries = [json.loads(l) for l in (beir / "queries.jsonl").read_text().splitlines()]
    assert len(queries) == 5 and all({"_id", "text"} <= q.keys() for q in queries)
    ids = {c["_id"] for c in corpus}
    assert all(l.split("\t")[1] in ids for l in lines[1:])                        # every qrel doc exists in the corpus
    no_abstract = sum(1 for p in paths for r in csv.DictReader(open(p, encoding="utf-8")) if not r["abstract_preview"].strip())
    assert sum(1 for c in corpus if not c["text"]) == no_abstract                 # preview mode fills text where one exists
    ex.export(paths, tmp_path, abstracts="none")
    assert all(json.loads(l)["text"] == "" for l in (beir / "corpus.jsonl").read_text().splitlines())






# ----------------------------------------------------------------- prepare / compare / import / review
def chat_judge(prepared_csv, title_to_label, out_csv, wrong_titles=(), conf="high"):
    """Simulates the judge file returned for a blinded input file."""
    rows = list(csv.DictReader(open(prepared_csv, encoding="utf-8")))
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "docid", "relevant", "confidence", "reason"])
        for r in rows:
            if not r["abstract"].strip():
                w.writerow([r["query_id"], r["docid"], "NA", "low", "no abstract"])
                continue
            lab = title_to_label[r["title"]]
            if r["title"] in wrong_titles:
                lab = 1 - lab
            w.writerow([r["query_id"], r["docid"], lab, conf, "test reason"])
    return rows


def _read(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _calibrated(sandbox, tmp_path):
    """Run prepare --calibration (Q1,Q2) + a perfect judge file + compare -> calibration.json has a PASSED 'judge1'."""
    data, t2l = sandbox
    prep = tmp_path / "cal_in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q1", "Q2", "--calibration", "--out", str(prep)))
    ans = tmp_path / "cal_out.csv"
    chat_judge(prep, t2l, ans)
    al.cmd_compare(_args("compare", "--file", str(ans)))
    return data, t2l


def test_prepare_is_blind_shuffled_and_strips_labels(sandbox, tmp_path):
    data, _ = sandbox
    out = tmp_path / "in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q1", "--calibration", "--out", str(out)))
    rows = _read(out)
    assert list(rows[0].keys()) == ["query_id", "query_text", "docid", "title", "abstract"]
    text = out.read_text(encoding="utf-8").lower()
    assert "tfidf" not in text and "biobert" not in text and "relevant" not in text
    _, orig = al.read_csv(data / "relevance_Q1.csv")
    assert {r["docid"] for r in rows} == {al.docid(r) for r in orig}
    assert [r["docid"] for r in rows] != [al.docid(r) for r in orig]             # order differs from rank order


def test_prepare_without_calibration_only_includes_unlabeled_rows(sandbox, tmp_path):
    data, _ = sandbox
    fields, rows = _unlabeled_copy(data)
    rows[0]["relevant"] = "1"
    al.write_csv_atomic(data / "relevance_Q6.csv", fields, rows)
    out = tmp_path / "in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(out)))
    assert len(_read(out)) == len(rows) - 1


def test_compare_pass_and_fail_and_gate_file(sandbox, tmp_path):
    data, t2l = sandbox
    _calibrated(sandbox, tmp_path)
    cal = json.loads(al.CALIBRATION_PATH.read_text())
    assert cal["judge1"]["passed"] and cal["judge1"]["kappa"] == 1.0
    prep = tmp_path / "p2.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q1", "Q2", "--calibration", "--out", str(prep)))
    bad = tmp_path / "bad.csv"
    chat_judge(prep, {t: 1 - v for t, v in t2l.items()}, bad)                      # systematically opposite judge
    with pytest.raises(SystemExit):
        al.cmd_compare(_args("compare", "--file", str(bad), "--judge", "badjudge"))
    assert json.loads(al.CALIBRATION_PATH.read_text())["badjudge"]["passed"] is False


def test_compare_rejects_incomplete_or_foreign_files(sandbox, tmp_path):
    data, t2l = sandbox
    prep = tmp_path / "p.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q1", "--calibration", "--out", str(prep)))
    ans = tmp_path / "a.csv"
    chat_judge(prep, t2l, ans)
    rows = _read(ans)
    short = tmp_path / "short.csv"
    with short.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows[:-2])
    with pytest.raises(SystemExit, match="missing"):
        al.cmd_compare(_args("compare", "--file", str(short)))
    rows[0]["docid"] = "pubmed:0000000"
    typo = tmp_path / "typo.csv"
    with typo.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit, match="not in"):
        al.cmd_compare(_args("compare", "--file", str(typo)))


def test_import_requires_passed_calibration(sandbox, tmp_path):
    data, t2l = sandbox
    _unlabeled_copy(data)
    prep = tmp_path / "in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    ans = tmp_path / "out.csv"
    chat_judge(prep, t2l, ans)
    with pytest.raises(SystemExit, match="PASSED calibration"):
        al.cmd_import(_args("import", "--file", str(ans)))


def test_import_writes_labels_notes_and_keeps_existing(sandbox, tmp_path):
    data, t2l = _calibrated(sandbox, tmp_path)
    fields, rows = _unlabeled_copy(data)
    rows[0]["relevant"] = str(1 - t2l[rows[0]["title"]])                          # pre-existing human label
    al.write_csv_atomic(data / "relevance_Q6.csv", fields, rows)
    prep = tmp_path / "in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    ans = tmp_path / "out.csv"
    chat_judge(prep, t2l, ans)
    al.cmd_import(_args("import", "--file", str(ans)))
    _, out = al.read_csv(data / "relevance_Q6.csv")
    assert out[0]["relevant"] == str(1 - t2l[out[0]["title"]])                    # untouched
    labeled = [r for r in out[1:] if al.has_text(r)]
    assert labeled and all(r["notes"].startswith("proposed:judge1 | conf=high | ") for r in labeled)
    assert all(r["relevant"] == str(t2l[r["title"]]) for r in labeled)
    assert list(out[0].keys()) == fields


@pytest.mark.parametrize("mutation,match", [
    ("drop_row", "missing"), ("bad_docid", "not in"), ("bad_value", "0, 1 or NA"), ("duplicate", "duplicate")])
def test_import_rejects_transcription_errors_and_writes_nothing(sandbox, tmp_path, mutation, match):
    data, t2l = _calibrated(sandbox, tmp_path)
    _unlabeled_copy(data)
    prep = tmp_path / "in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    ans = tmp_path / "out.csv"
    chat_judge(prep, t2l, ans)
    rows = _read(ans)
    if mutation == "drop_row": rows = rows[1:]
    if mutation == "bad_docid": rows[0]["docid"] += "9"
    if mutation == "bad_value": rows[0]["relevant"] = "maybe"
    if mutation == "duplicate": rows.append(dict(rows[0]))
    bad = tmp_path / "bad.csv"
    with bad.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    before = (data / "relevance_Q6.csv").read_bytes()
    with pytest.raises(SystemExit, match=match):
        al.cmd_import(_args("import", "--file", str(bad)))
    assert (data / "relevance_Q6.csv").read_bytes() == before                       # nothing written on failure


def test_import_forces_NA_for_papers_without_abstract(sandbox, tmp_path):
    data, t2l = _calibrated(sandbox, tmp_path)
    fields, rows = _unlabeled_copy(data)
    rows[0]["abstract_preview"] = ""
    al.write_csv_atomic(data / "relevance_Q6.csv", fields, rows)
    prep = tmp_path / "in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    ans = tmp_path / "out.csv"
    chat_judge(prep, t2l, ans)
    al.cmd_import(_args("import", "--file", str(ans)))
    _, out = al.read_csv(data / "relevance_Q6.csv")
    assert out[0]["relevant"] == "" and out[0]["notes"].startswith("NEEDS_HUMAN")
    rows2 = _read(ans)
    for r in rows2:
        if r["docid"] == al.docid(rows[0]):
            r["relevant"] = "1"
    guess = tmp_path / "guess.csv"
    with guess.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows2[0].keys()); w.writeheader(); w.writerows(rows2)
    al.write_csv_atomic(data / "relevance_Q6.csv", fields, rows)                    # reset
    with pytest.raises(SystemExit, match="must answer NA"):
        al.cmd_import(_args("import", "--file", str(guess)))


def test_review_records_overrides_and_provenance(sandbox, tmp_path):
    data, t2l = _calibrated(sandbox, tmp_path)
    _unlabeled_copy(data)
    prep = tmp_path / "in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    ans = tmp_path / "out.csv"
    chat_judge(prep, t2l, ans, conf="medium")
    al.cmd_import(_args("import", "--file", str(ans)))
    al.cmd_review(_args("review", "make"))
    sheet = al.REVIEW_DIR / "review_sheet.csv"
    rows = _read(sheet)
    assert rows[0]["proposed_confidence"] == "medium" and rows[0]["proposed_reason"] == "test reason"
    with pytest.raises(SystemExit, match="review everything"):
        al.cmd_review(_args("review", "apply"))                                    # nothing entered yet
    for i, r in enumerate(rows):
        r["human_label"] = r["proposed_label"] if r["proposed_label"] in ("0", "1") else "1"
    flip = next(r for r in rows if r["proposed_label"] in ("0", "1"))
    flip["human_label"] = str(1 - int(flip["proposed_label"]))                          # one deliberate override
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    al.cmd_review(_args("review", "apply"))
    res = json.loads((al.REVIEW_DIR / "review_result.json").read_text())
    assert res["overridden"] == 1 and res["rows_reviewed"] == len(rows)
    _, out = al.read_csv(data / "relevance_Q6.csv")
    assert all(r["notes"].startswith("reviewed: ") for r in out if r["notes"])
    fl = next(r for r in out if al.docid(r) == flip["docid"])
    assert fl["relevant"] == flip["human_label"]
    prov = {}
    for r in out:
        k = ex.provenance_of(r); prov[k] = prov.get(k, 0) + 1
    assert prov == {"author_reviewed": len(out)}


def test_audit_is_blind_and_scores_against_separate_key(sandbox, tmp_path):
    data, t2l = _calibrated(sandbox, tmp_path)
    _unlabeled_copy(data)
    prep = tmp_path / "in.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    ans = tmp_path / "out.csv"
    chat_judge(prep, t2l, ans)
    al.cmd_import(_args("import", "--file", str(ans)))
    al.cmd_audit(_args("audit", "make", "--fraction", "0.5"))
    rows = _read(al.AUDIT_PATH)
    assert rows and "proposed_label" not in rows[0] and all(r["human_label"] == "" for r in rows)
    key = {(r["query_id"], r["docid"]): r["proposed_label"] for r in _read(al.REVIEW_DIR / "audit_key.csv")}
    for r in rows:
        r["human_label"] = key[(r["query_id"], r["docid"])]
    with al.AUDIT_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    al.cmd_audit(_args("audit", "score"))
    assert json.loads((al.REVIEW_DIR / "audit_result.json").read_text())["agreement"] == 1.0


def test_provenance_classification():
    assert ex.provenance_of({"notes": "proposed:judge1 | conf=high | x"}) == "proposed_unreviewed"
    assert ex.provenance_of({"notes": "NEEDS_HUMAN judge_abstained | x"}) == "author_labeled_after_abstain"
    assert ex.provenance_of({"notes": "checked twice"}) == "human"
    assert ex.provenance_of({"notes": "reviewed: proposed=1 final=0"}) == "author_reviewed"


def test_export_cli_refuses_unreviewed_labels(tmp_path):
    import subprocess, sys
    rows = _read(ROOT / "data" / "relevance_Q1.csv")
    rows[0]["notes"] = "proposed:judge1 | conf=high | x"
    bad = tmp_path / "relevance_Q1.csv"
    with bad.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    cmd = [sys.executable, str(ROOT / "scripts" / "export_benchmark.py"), "--glob", str(bad), "--out", str(tmp_path / "o")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "unreviewed" in (r.stdout + r.stderr)
    ok = subprocess.run(cmd + ["--allow-unreviewed"], capture_output=True, text=True, cwd=ROOT)
    assert ok.returncode == 0


def test_flagged_only_review_and_accept_flow(sandbox, tmp_path):
    data, t2l = _calibrated(sandbox, tmp_path)
    _unlabeled_copy(data)
    prep, ans = tmp_path / "in.csv", tmp_path / "out.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    chat_judge(prep, t2l, ans)
    al.cmd_import(_args("import", "--file", str(ans)))
    with pytest.raises(SystemExit):                                   # no audit yet -> refuse
        al.cmd_accept(_args("accept"))
    fields, rows = al.read_csv(data / "relevance_Q6.csv")
    rows[0]["notes"] = "proposed:judge1 | conf=low | unsure"        # one flagged row
    al.write_csv_atomic(data / "relevance_Q6.csv", fields, rows)
    al.cmd_review(_args("review", "make", "--flagged-only"))
    sheet = al.REVIEW_DIR / "review_sheet.csv"
    srows = _read(sheet)
    assert len(srows) == 1 + sum(1 for r in rows if r["notes"].startswith("NEEDS_HUMAN"))
    for r in srows:
        r["human_label"] = "0"
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=srows[0].keys()); w.writeheader(); w.writerows(srows)
    al.cmd_review(_args("review", "apply"))
    al.cmd_audit(_args("audit", "make", "--fraction", "0.5"))
    arows = _read(al.AUDIT_PATH)
    key = {(r["query_id"], r["docid"]): r["proposed_label"] for r in _read(al.REVIEW_DIR / "audit_key.csv")}
    for i, r in enumerate(arows):
        k = key[(r["query_id"], r["docid"])]
        r["human_label"] = ("1" if k == "0" else "0") if i == 0 else k    # author disagrees on the first sampled row
    with al.AUDIT_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=arows[0].keys()); w.writeheader(); w.writerows(arows)
    al.cmd_audit(_args("audit", "score"))
    al.cmd_accept(_args("accept", "--min-audit", "1"))
    final = _read(data / "relevance_Q6.csv")
    assert not any(r["notes"].startswith(("proposed:", "NEEDS_HUMAN")) for r in final)
    first = next(r for r in final if (r["query_id"], al.docid(r)) == (arows[0]["query_id"], arows[0]["docid"]))
    assert first["relevant"] == arows[0]["human_label"] and first["notes"].startswith("reviewed:")
    assert any(r["notes"].startswith("accepted:") for r in final)
    assert ex.provenance_of({"notes": "accepted: proposed=1 final=1"}) == "proposed_audited_accepted"


def test_files_saved_with_a_byte_order_mark_are_read(sandbox, tmp_path):
    """Excel's 'CSV UTF-8' and some editors prepend a BOM; the first column must still be recognised."""
    data, t2l = _calibrated(sandbox, tmp_path)
    _unlabeled_copy(data)
    prep, ans = tmp_path / "in.csv", tmp_path / "out.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    chat_judge(prep, t2l, ans)
    al.cmd_import(_args("import", "--file", str(ans)))
    al.cmd_audit(_args("audit", "make", "--fraction", "0.5"))
    key = {(r["query_id"], r["docid"]): r["proposed_label"] for r in _read(al.REVIEW_DIR / "audit_key.csv")}
    rows = _read(al.AUDIT_PATH)
    for r in rows:
        r["human_label"] = key[(r["query_id"], r["docid"])]
    import io
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    al.AUDIT_PATH.write_bytes(b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))   # saved the way Excel does
    al.cmd_audit(_args("audit", "score"))
    assert json.loads((al.REVIEW_DIR / "audit_result.json").read_text())["agreement"] == 1.0
    ex.load_query_file  # exporter must also tolerate a BOM in a pooled CSV
    bom = tmp_path / "relevance_Q9.csv"
    src = (data / "relevance_Q1.csv").read_bytes()
    bom.write_bytes(b"\xef\xbb\xbf" + src)
    assert ex.load_query_file(str(bom))[0] == "Q1"


def test_accept_can_label_abstract_less_papers_zero_by_rule(sandbox, tmp_path):
    data, t2l = _calibrated(sandbox, tmp_path)
    fields, rows = _unlabeled_copy(data)
    for r in rows[:2]:
        r["abstract_preview"] = ""
    al.write_csv_atomic(data / "relevance_Q6.csv", fields, rows)
    prep, ans = tmp_path / "in.csv", tmp_path / "out.csv"
    al.cmd_prepare(_args("prepare", "--queries", "Q6", "--out", str(prep)))
    chat_judge(prep, t2l, ans)
    al.cmd_import(_args("import", "--file", str(ans)))
    al.cmd_audit(_args("audit", "make", "--fraction", "0.5"))
    key = {(r["query_id"], r["docid"]): r["proposed_label"] for r in _read(al.REVIEW_DIR / "audit_key.csv")}
    arows = _read(al.AUDIT_PATH)
    for r in arows:
        r["human_label"] = key[(r["query_id"], r["docid"])]
    with al.AUDIT_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=arows[0].keys()); w.writeheader(); w.writerows(arows)
    al.cmd_audit(_args("audit", "score"))
    with pytest.raises(SystemExit):                                      # without the flag, abstained rows block accept
        al.cmd_accept(_args("accept", "--min-audit", "1"))
    al.cmd_accept(_args("accept", "--min-audit", "1", "--no-abstract-as-zero"))
    final = _read(data / "relevance_Q6.csv")
    zero = [r for r in final if r["notes"].startswith("rule:")]
    assert len(zero) == 2 and all(r["relevant"] == "0" and not r["abstract_preview"].strip() for r in zero)
    assert ex.provenance_of({"notes": "rule: no abstract -> not relevant"}) == "no_abstract_rule"
