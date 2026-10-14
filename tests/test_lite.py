"""G1-lite pieces: --no-downstream, pilot-format G0 inputs with a shared sample, test_id subset, run_lite.sh syntax,
lite stats report. Mock / fake data only, no network, no real sealed files."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import make_test_subset as mts
import seal
import stats
import validate_judge as vj
from run import run_condition
from test_e2e_mock import _inst

ROOT = Path(__file__).resolve().parent.parent


def test_no_downstream_records(tmp_path, make_client):
    data = tmp_path / "mini.jsonl"
    rows = [_inst(1, "misinformation", "1990", "1995"), _inst(3, "none", "1970")]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    client = make_client("mock")
    recs = run_condition("C0", "dev", 0.3, client, results_dir=tmp_path / "out", data_path=data, downstream=False)
    for r in recs:
        assert r["downstream"]["skipped"] is True
        assert all(r["downstream"][k] is None for k in ("answer", "D", "category", "committed_value", "model"))
    assert recs[0]["csr"] in (0, 1) and isinstance(recs[1]["false_conflict"], bool)
    assert client.summary()["calls_total"] == 4  # compress + judge per instance, no downstream / D judge


def _pilot(tmp_path):
    dev = [{"id": f"dev_{i:04d}", "question": f"q{i}", "conflict_type": "wiki_natural",
            "gold": {"answer_a": f"{1900 + i}", "answer_b": f"{1800 + i}", "span_a": f"{1900 + i}",
                     "span_b": f"{1800 + i}"}} for i in range(12)]
    vj.write_jsonl(tmp_path / "dev.jsonl", dev)
    files = []
    for cond in ("C0", "C1"):
        p = tmp_path / f"m_{cond}.jsonl"
        vj.write_jsonl(p, [{"id": d["id"], "model": "m:1", "condition": cond, "r": 0.3, "budget": 50,
                            "out_tokens": 5, "over_budget": False, "retried": False,
                            "text": f"{cond} says {d['gold']['answer_a']}"} for d in dev])
        files.append(str(p))
    return files


def test_pilot_join_and_shared_sample(tmp_path, monkeypatch, make_client):
    files = _pilot(tmp_path)
    monkeypatch.setattr(vj, "DEV_FILE", tmp_path / "dev.jsonl")
    monkeypatch.setattr(vj, "make_client", lambda mode, max_cost: make_client(mode))
    out = tmp_path / "g0_local"
    common = ["--results", *files, "--n", "8", "--out-dir", str(out)]

    vj.main(["label-sheet", *common])
    vj.main(["negative-control", *common])
    for j in ("opt", "eval"):
        vj.main(["rejudge", "--judge", j, "--mode", "mock", *common])
        vj.main(["negative-control", "--run", "--judge", j, "--mode", "mock", "--out-dir", str(out)])

    sample = [tuple(k) for k in json.loads((out / "sample_ids.json").read_text())["items"]]
    assert len(sample) == 8 and [c for c, _ in sample].count("C0") == 4
    assert len({i for _, i in sample}) == 8  # distinct instances across conditions
    key = json.loads((out / "label_key.json").read_text())["map"]
    from_sheet = {(v["condition"], v["id"]) for v in key.values()}
    from_rejudge = {(r["condition"], r["id"]) for r in vj.load_records([out / "rejudge_opt.jsonl"])}
    from_neg = {(r["condition"], r["id"]) for r in vj.load_records([out / "negative_control_items.jsonl"])}
    assert set(sample) == from_sheet == from_rejudge == from_neg
    item = vj.load_records([out / "negative_control_items.jsonl"])[0]
    assert item["question"].startswith("q") and item["compressed"].startswith(item["condition"])

    with pytest.raises(SystemExit):  # another n against the saved sample
        vj.main(["rejudge", "--judge", "opt", "--mode", "mock", "--results", *files, "--n", "4",
                 "--out-dir", str(out)])

    with pytest.raises(SystemExit) as e:  # labels still empty
        vj.main(["agreement", "--dir", str(out)])
    assert e.value.code == vj.UNLABELLED_EXIT
    labels = out / "author_labels.csv"
    with open(labels, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(a_supported="1", a_distorted="0", b_supported="0", b_distorted="0")
    with open(labels, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(SystemExit):  # filled labels are never overwritten
        vj.main(["label-sheet", *common])
    vj.main(["agreement", "--dir", str(out)])
    res = json.loads((out / "agreement.json").read_text())
    assert isinstance(res["pass"], bool) and set(res["checks"]) == {
        "accuracy_opt", "accuracy_eval", "neg_fp_opt", "neg_fp_eval", "inter_judge_kappa"}
    assert res["report"]["n_labelled"] == 8


def test_make_test_subset(tmp_path):
    sealed = tmp_path / "data" / "sealed"
    sealed.mkdir(parents=True)
    pool = sealed / "test_id_pool.jsonl"
    pool.write_bytes(b"".join(f'{{"id": "p{i}"}}\n'.encode() for i in range(10)))
    (sealed / "test_ood2.jsonl").write_bytes(b'{"id": "o"}\n')
    seal.seal_files([pool, sealed / "test_ood2.jsonl"], tmp_path)

    out = mts.make_test_id(4, tmp_path)
    head = b"".join(pool.read_bytes().splitlines(keepends=True)[:4])
    assert out.read_bytes() == head
    latest = seal._latest_seal(tmp_path)
    assert latest["data/sealed/test_id.jsonl"] == hashlib.sha256(head).hexdigest()
    assert {"data/sealed/test_id_pool.jsonl", "data/sealed/test_ood2.jsonl"} <= set(latest)
    seal.verify_sealed(tmp_path)

    md = (tmp_path / "data" / "SEALED.md").read_text()
    mts.make_test_id(4, tmp_path)  # same content: passes, no new seal section
    assert (tmp_path / "data" / "SEALED.md").read_text() == md
    with pytest.raises(seal.SealError):  # existing file differs from the pool head
        mts.make_test_id(5, tmp_path)
    with pytest.raises(SystemExit):
        mts.main(["--split", "test_ood1", "--n", "100"])


def test_run_lite_syntax():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "run_lite.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_lite_report(tmp_path):
    rates = {"C0": 0.3, "C1": 0.5, "C2": 0.6, "C3-G1": 0.7, "C3-G": 0.8}
    for c, p in rates.items():
        for sp, n in (("test_id", 40), ("test_ood1", 20), ("test_ood2", 20), ("distractor", 10)):
            if c == "C2" and sp == "test_ood2":
                continue  # a missing result file is reported, not an error
            k = int(n * (p - 0.2 if sp != "test_id" else p))
            recs = []
            for i in range(n):
                if sp == "distractor":
                    recs.append({"id": f"{sp}{i}", "csr": None, "false_conflict": i < 2, "over_budget": False,
                                 "downstream": {"D": None, "skipped": True}})
                else:
                    recs.append({"id": f"{sp}{i}", "csr": int(i < k), "false_conflict": None, "over_budget": i == 0,
                                 "downstream": {"D": int(i % 2 == 0)} if sp == "test_id" else {"D": None}})
            vj.write_jsonl(tmp_path / f"{c}_{sp}_r0.3_s0.jsonl", recs)
    md = stats.lite_report(tmp_path, n_boot=300)
    for s in ("예비", "Holm", "| H1 | C1 − C0 | 40 | 20 | 12 |", "| H2a |", "| H2b |", "## 3. H3", "H4-test_ood1",
              "H4-test_ood2", "| test_ood2 | C2 | 결과 없음", "| distractor | C0 | 10 |", "분산"):
        assert s in md, s
    assert "2/10" in md and "20/40" in md  # FCR and D counts


def test_lite_report_multi_opt(tmp_path):
    """Phi-4 run: C3-P/C3-M replace C3-G1/C3-G; H3/H4 are reported per optimized condition."""
    for c, p in {"C0": 0.3, "C1": 0.5, "C2": 0.6, "C3-P": 0.7, "C3-M": 0.8}.items():
        for sp, n in (("test_id", 40), ("test_ood1", 20), ("test_ood2", 20)):
            k = int(n * p)
            vj.write_jsonl(tmp_path / f"{c}_{sp}_r0.3_s0.jsonl",
                           [{"id": f"{sp}{i}", "csr": int(i < k), "downstream": {"D": None}} for i in range(n)])
    md = stats.lite_report(tmp_path, n_boot=300, conds=["C0", "C1", "C2", "C3-P", "C3-M"],
                           paired=[("H1", "C1", "C0", 0.10), ("H2-P", "C3-P", "C2", 0.05)],
                           opt_conds=["C3-P", "C3-M"], label="Phi-4")
    for s in ("# Phi-4 예비 결과", "| test_id | C3-M | 40 | 32/40", "| H2-P | C3-P − C2 |", "- C3-P: n = 40",
              "- C3-M: n = 40", "| H4-C3-M-test_ood2 |", "H3-C3-P, H3-C3-M"):
        assert s in md, s
    assert "C3-G" not in md
