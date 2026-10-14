"""Amendment 2.2-d: G0 split (label-free checks first, author accuracy last), labelling server, report header."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import label_server as ls
import stats
import validate_judge as vj

ROOT = Path(__file__).resolve().parent.parent


def _judge(csr: int) -> dict:
    return {"a_supported": True, "a_distorted": False, "b_supported": bool(csr), "b_distorted": False}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _rejudge(path: Path, csrs: list[int]) -> Path:
    return _write(path, [{"condition": "C0" if k % 2 else "C1", "id": f"x{k}", "judge": _judge(c)}
                         for k, c in enumerate(csrs)])


def _neg(path: Path, fp: int, n: int = 50) -> Path:
    return _write(path, [{"judge": {"b_supported": k < fp}} for k in range(n)])


def test_judge_only_checks_pass_and_fail(tmp_path):
    csrs = [1] * 30 + [0] * 20
    rej = {"opt": _rejudge(tmp_path / "rejudge_opt.jsonl", csrs), "eval": _rejudge(tmp_path / "rejudge_eval.jsonl", csrs)}
    neg = {"opt": _neg(tmp_path / "neg_opt.jsonl", 2), "eval": _neg(tmp_path / "neg_eval.jsonl", 5)}   # FP 0.04, 0.10
    res = vj.judge_only_checks(rej, neg)
    assert res["pass"] is True and res["report"]["inter_judge"]["kappa"] == pytest.approx(1.0)
    neg["eval"] = _neg(tmp_path / "neg_eval.jsonl", 6)                                                  # FP 0.12
    assert vj.judge_only_checks(rej, neg)["checks"]["neg_fp_eval"] is False
    rej["eval"] = _rejudge(tmp_path / "rejudge_eval.jsonl", [1 - c for c in csrs])                     # judges disagree
    assert vj.judge_only_checks(rej, neg)["checks"]["inter_judge_kappa"] is False
    assert vj.judge_only_checks({"opt": rej["opt"]}, neg)["pass"] is False                              # one judge missing


def test_judge_checks_cli_writes_g0_auto(tmp_path):
    csrs = [1] * 30 + [0] * 20
    for n in ("opt", "eval"):
        _rejudge(tmp_path / f"rejudge_{n}.jsonl", csrs)
        _neg(tmp_path / f"negative_control_{n}.jsonl", 1)
    vj.main(["judge-checks", "--dir", str(tmp_path)])
    assert json.loads((tmp_path / "g0_auto.json").read_text())["pass"] is True
    _neg(tmp_path / "negative_control_opt.jsonl", 20)
    with pytest.raises(SystemExit):
        vj.main(["judge-checks", "--dir", str(tmp_path)])


# ---------- labelling server ----------
def _sheet(tmp_path) -> Path:
    p = tmp_path / "author_labels.csv"
    fields = ["item_key", "question", "compressed", "answer_a", "answer_b", *ls.FLAGS, "note"]
    rows = [{"item_key": f"k{n}", "question": f"Q{n}?", "compressed": f"[d0] <b>{n}</b>", "answer_a": "1990",
             "answer_b": "1995", **{f: "" for f in ls.FLAGS}, "note": ""} for n in range(3)]
    ls.write_rows(p, fields, rows)
    return p


def test_label_rows_roundtrip_and_validation(tmp_path):
    p = _sheet(tmp_path)
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")                                  # UTF-8 BOM like label-sheet
    fields, rows = ls.load_rows(p)
    ls.apply_update(rows, {"item_key": "k1", "a_supported": "1", "a_distorted": "0", "note": "체크"})
    with pytest.raises(ValueError):
        ls.apply_update(rows, {"item_key": "k1", "b_supported": "yes"})
    with pytest.raises(KeyError):
        ls.apply_update(rows, {"item_key": "nope", "a_supported": "1"})
    ls.write_rows(p, fields, rows)
    _, again = ls.load_rows(p)
    assert again[1]["a_supported"] == "1" and again[1]["note"] == "체크" and ls.labelled(again) == 0
    vj_done, vj_empty = vj.count_labels(p)                                             # validate_judge reads it
    assert (vj_done, vj_empty) == (0, 3)


def test_label_server_http_saves_to_csv(tmp_path):
    p = _sheet(tmp_path)
    fields, rows = ls.load_rows(p)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ls.make_handler(p, fields, rows, threading.Lock()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "G0 저자 라벨" in page and "label_key.json은 열지 마세요" in page
        items = json.loads(urllib.request.urlopen(base + "/items").read())
        assert [x["item_key"] for x in items] == ["k0", "k1", "k2"]
        for k in ("k0", "k1", "k2"):
            body = json.dumps({"item_key": k, **{f: "1" if "supported" in f else "0" for f in ls.FLAGS}}).encode()
            req = urllib.request.Request(base + "/save", data=body, headers={"Content-Type": "application/json"})
            assert urllib.request.urlopen(req).read() == b"ok"
        bad = urllib.request.Request(base + "/save", data=json.dumps({"item_key": "k0", "a_supported": "2"}).encode())
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        srv.shutdown()
    assert vj.count_labels(p) == (3, 0)                                                # complete for `agreement`


def test_label_server_never_reads_the_key():
    src = (ROOT / "src" / "label_server.py").read_text(encoding="utf-8")
    code = src.split('PAGE = r"""')[0] + src.split('</html>"""')[1]                    # outside the page text
    assert "label_key" not in code.replace("never opens label_key.json", "")


# ---------- report header and pipeline wiring ----------
def test_g0_status_line(tmp_path):
    assert "미실행" in stats.g0_status_line(tmp_path) and "미확정" in stats.g0_status_line(tmp_path)
    (tmp_path / "g0_auto.json").write_text(json.dumps({"pass": True}))
    line = stats.g0_status_line(tmp_path)
    assert "라벨 없는 기준(κ·음성 대조) 통과" in line and "미확정" in line
    (tmp_path / "agreement.json").write_text(json.dumps({"pass": True}))
    assert "미확정" not in stats.g0_status_line(tmp_path)


def test_run_lite_runs_label_free_g0_first_and_author_g0_last():
    sh = (ROOT / "scripts" / "run_lite.sh").read_text(encoding="utf-8")
    assert "phases=(preflight g0auto gepa gate freeze test stats)" in sh
    for phase in ("gepa", "gate", "freeze", "test"):
        body = sh.split(f"phase_{phase}() {{")[1].split("\n}")[0]
        assert f"require_g0auto {phase}" in body and "require_g0 " not in body
    assert "judge-checks" in sh.split("phase_g0auto() {")[1].split("\n}")[0]
    assert "agreement" in sh.split("phase_g0() {")[1].split("\n}")[0]
    assert "label_server.py" in sh.split('if [[ "$1" == all ]]; then')[1]
