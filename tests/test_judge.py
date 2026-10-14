import csv
import json

import pytest

import validate_judge as vj
from judge import (CCDJudge, CSRJudge, DJudge, DownstreamGenerator, FCRJudge, load_prompt,
                   parse_json_reply, span_present)


class FakeClient:
    """Same signature as LLMClient.call; returns scripted replies, records calls."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def call(self, model, system, prompt, max_tokens, temperature=0.0):
        self.calls.append({"model": model, "system": system, "prompt": prompt})
        return (self.replies.pop(0) if self.replies else ""), {"mock": True}


INST = {"id": "dev_x", "question": "What year did X die?",
        "gold": {"answer_a": "1860", "answer_b": "1846", "span_a": "1860", "span_b": "1846"}}
ALL_OK = json.dumps({"a_supported": True, "a_distorted": False, "b_supported": True, "b_distorted": False,
                     "conflict_signaled": True, "rationale": "both listed"})


def test_revised_csr_prompt_loaded():
    p = load_prompt("judge_csr.txt")
    assert "does not explicitly appear in the CONTEXT, supported = false" in p
    assert "NONE of the following is a distortion" in p
    assert CSRJudge(FakeClient(), "m").system_prompt == p


def test_over_budget_skips_call():
    c = FakeClient([ALL_OK])
    r = CSRJudge(c, "m").evaluate(INST, "X died in 1860 or 1846.", over_budget=True)
    assert c.calls == [] and r["csr"] == 0 and r["meta"]["judge_skipped"]
    f = FCRJudge(c, "m").evaluate(INST, "x", over_budget=True)
    assert c.calls == [] and f["false_conflict"] is False


@pytest.mark.parametrize("reply,method", [
    (ALL_OK, "json"),
    ("```json\n" + ALL_OK + "\n```", "json"),
    ("Here you go: " + ALL_OK + " done", "json"),
    ('{"a_supported": true, "a_distorted": false, "b_supported": true, "b_distorted": false, '
     '"conflict_signaled": true, "rationale": "unterminated', "regex"),
])
def test_csr_parsing(reply, method):
    r = CSRJudge(FakeClient([reply]), "m").evaluate(INST, "X died in 1860; others say 1846.", False)
    assert r["parse_ok"] and r["parse_method"] == method
    assert r["csr"] == 1 and r["csr_lenient"] == 1 and r["csr_strict"] == 1


def test_csr_parse_failure_counted():
    r = CSRJudge(FakeClient(["I think both are fine."]), "m").evaluate(INST, "1860 1846", False)
    assert r["parse_ok"] is False and r["csr"] == 0 and r["a_supported"] is False


def test_csr_strict_value_matching():
    assert span_present("20 acres (81,000 m2)", "covers 20 acres (81,000 m²) in total")
    assert not span_present("20 acres (81,000 m2)", "covers 20.5 acres (83,000 m²)")
    assert span_present("27 August 1968", "on August 27, 1968 the")
    r = CSRJudge(FakeClient([ALL_OK]), "m").evaluate(INST, "X died in 1860.", False)
    assert r["csr"] == 1 and r["csr_strict"] == 0      # 1846 missing from text


@pytest.mark.parametrize("reply,cat,val,d,ok", [
    ('{"category": "CONFLICT_ACK", "committed_value": "NONE"}', "CONFLICT_ACK", "NONE", 1, True),
    ('```json\n{"category": "definitive", "committed_value": "a"}\n```', "DEFINITIVE", "A", 0, True),
    ('{"category": "INSUFFICIENT", "committed_value": "zzz"}', "INSUFFICIENT", "OTHER", 0, True),
    ("no json here", "OTHER", "NONE", 0, False),
])
def test_d_judge_mapping(reply, cat, val, d, ok):
    c = FakeClient([reply])
    r = DJudge(c, "m").evaluate(INST, "It is 1860 or 1846.")
    assert (r["category"], r["committed_value"], r["d"], r["parse_ok"]) == (cat, val, d, ok)
    assert c.calls[0]["system"] == "You are a strict annotator of question-answering system outputs."
    assert 'A = "1860", B = "1846"' in c.calls[0]["prompt"] and '{"category"' in c.calls[0]["prompt"]


def test_downstream_and_fcr():
    c = FakeClient(["It differs: 1860 or 1846.", '{"false_conflict": true, "rationale": "two years"}'])
    assert DownstreamGenerator(c, "m").answer(INST, "ctx") == "It differs: 1860 or 1846."
    assert FCRJudge(c, "m").evaluate(INST, "ctx", False)["false_conflict"] is True


def test_ccd_not_implemented():
    with pytest.raises(NotImplementedError):
        CCDJudge(FakeClient(), "m").evaluate(INST, "orig", "comp")


def _recs():
    out = []
    for cond in ("C0", "C1"):
        for i, (a, b) in enumerate([("1860", "1846"), ("Shiraz", "Herat"), ("Yes", "No"), ("42", "41"),
                                    ("The Daily Worker", "The Hoover Gazette")]):
            out.append({"id": f"dev_{i}", "condition": cond, "question": f"q{i}", "over_budget": False,
                        "compressed": f"text {i} mentions {a}", "gold": {"answer_a": a, "answer_b": b}})
    return out


def test_label_sheet_is_blind(tmp_path):
    rows, key = vj.build_label_sheet(_recs(), seed=7)
    path = tmp_path / "s.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = path.read_text()
    assert "C0" not in text and "C1" not in text and "condition" not in text
    assert len(rows) == 10 and len(key) == 10
    assert [r["item_key"] for r in rows] == [r["item_key"] for r in vj.build_label_sheet(_recs(), 7)[0]]
    assert {k["condition"] for k in key.values()} == {"C0", "C1"}


def test_negative_control_uses_foreign_answer():
    recs = _recs()
    items = vj.build_negative_items(recs, seed=7)
    assert len(items) == len(recs)
    for it, r in zip(items, recs):
        assert it["donor_id"] != r["id"]
        assert it["gold"]["answer_b"] not in (r["gold"]["answer_a"], r["gold"]["answer_b"])
        assert it["gold"]["answer_b"].lower() not in ("yes", "no")
        assert it["gold"]["answer_a"] == r["gold"]["answer_a"]


def test_agreement_end_to_end(tmp_path):
    recs = _recs()
    rows, key = vj.build_label_sheet(recs, seed=7)
    for r in rows:
        r.update(a_supported="1", a_distorted="0", b_supported="1", b_distorted="0")
    labels = tmp_path / "l.csv"
    with open(labels, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keyp = tmp_path / "k.json"
    keyp.write_text(json.dumps({"map": key}))
    flags = dict(a_supported=True, a_distorted=False, b_supported=True, b_distorted=False, parse_ok=True)
    for name in ("opt", "eval"):
        vj.write_jsonl(tmp_path / f"{name}.jsonl", [{"id": r["id"], "condition": r["condition"], "judge": flags}
                                                     for r in recs])
    rep = vj.compute_agreement(labels, keyp, {n: tmp_path / f"{n}.jsonl" for n in ("opt", "eval")})
    assert rep["n_labelled"] == 10 and rep["judges"]["opt"]["csr_accuracy"] == 1.0
    assert "PASS" in vj.render_agreement(rep)
