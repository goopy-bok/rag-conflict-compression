"""C3 APO layer: split, feedback, score, DSPy client bridge, GEPA/MIPROv2 mock runs, few-shot check,
C3' variants, prompt freeze. No network: litellm entry points and socket connects raise."""
import json
import os
import re
import socket

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no price-map download on import
import litellm  # noqa: E402  must be imported before dspy (a later import hits dspy 3.4's lazy-import cycle)
import pytest  # noqa: E402

import apo_common as ac  # noqa: E402
import dspy_adapter as da  # noqa: E402
from budget import count_tokens
from compressors.llm import PromptCompressor

MODEL = "openai/gpt-4o-mini-2024-07-18"
pytestmark = pytest.mark.filterwarnings("ignore:Implementing custom LMs through BaseLM.forward:DeprecationWarning")


# ------------------------------------------------------------------ fixtures

def _inst(i, ctype="misinformation", a="1990", b="1995"):
    docs = [{"doc_id": f"d{k}", "text": f"Document {k} about item {i}. " * 15, "role": "distractor"} for k in range(3)]
    gold = {"answer_a": a, "span_a": a}
    docs[2].update(text=f"Item {i} was founded in {a}. " * 8, role="conflict_a" if ctype != "none" else "default")
    if ctype != "none":
        docs[0].update(text=f"Item {i} was founded in {b}. " * 8, role="conflict_b")
        gold.update(answer_b=b, span_b=b)
    return {"id": f"{'x' if ctype == 'none' else 'c'}{i}", "source": "conflictbank", "conflict_type": ctype,
            "question": f"When was item {i} founded?", "docs": docs, "gold": gold,
            "conflict_positions": [0, 2] if ctype != "none" else [], "entity": f"item {i}",
            "total_tokens": sum(count_tokens(d["text"]) for d in docs)}


ROWS = {"dev": [_inst(i) for i in range(297)], "dev_distractor": [_inst(1000 + i, "none") for i in range(100)]}


def _small_split():
    return {"train": [(_inst(i), 0.3 if i % 2 else 0.5) for i in range(5)] + [(_inst(900, "none"), 0.5)],
            "val": [(_inst(10 + i), 0.5) for i in range(3)] + [(_inst(901, "none"), 0.3)]}


@pytest.fixture
def no_network(monkeypatch):
    from dspy.clients.engines import litellm_engine

    def boom(*a, **k):
        raise AssertionError("litellm / network must not be used")
    for name in ("completion", "acompletion", "text_completion", "atext_completion", "responses", "aresponses"):
        monkeypatch.setattr(litellm, name, boom)
    for cls in (litellm_engine.LiteLLMEngine, litellm_engine.AsyncLiteLLMEngine):
        for name in ("complete", "complete_legacy"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, boom)
    monkeypatch.setattr(socket.socket, "connect", boom)


# ------------------------------------------------------------------ split

def test_apo_split_deterministic_disjoint_sizes(tmp_path):
    s1 = ac.build_apo_split(0, tmp_path / "a.json", ROWS)
    s2 = ac.build_apo_split(0, tmp_path / "b.json", ROWS)
    assert s1 == s2
    tr, va = {e["id"] for e in s1["train"]}, {e["id"] for e in s1["val"]}
    assert not tr & va and len(tr) == 247 and len(va) == 150
    assert sum(i.startswith("x") for i in tr) == 50 and sum(i.startswith("x") for i in va) == 50
    for part in ("train", "val"):
        rs = [e["r"] for e in s1[part]]
        assert set(rs) == {0.3, 0.5} and abs(rs.count(0.3) - rs.count(0.5)) <= 2
    assert ac.build_apo_split(1, tmp_path / "c.json", ROWS) != s1


def test_apo_split_is_fixed_once_written(tmp_path):
    p = tmp_path / "s.json"
    ac.build_apo_split(0, p, ROWS)
    saved = json.loads(p.read_text())
    saved["val"] = saved["val"][:3]
    p.write_text(json.dumps(saved))
    assert ac.build_apo_split(0, p, ROWS)["val"] == saved["val"]  # read, never rebuilt
    with pytest.raises(ValueError):
        ac.build_apo_split(1, p, ROWS)
    loaded = ac.load_apo_split(0, p, ROWS)
    assert len(loaded["val"]) == 3 and isinstance(loaded["val"][0][0], dict) and loaded["val"][0][1] in (0.3, 0.5)


# ------------------------------------------------------------------ feedback and score

def test_describe_loss_names_answers_and_sources_without_ccd_types():
    inst = _inst(1)
    texts = [
        ac.describe_loss(inst, {"a_supported": False, "b_supported": True}, False, 50, 100),
        ac.describe_loss(inst, {"a_supported": True, "b_supported": True, "b_distorted": True}, False, 50, 100),
        ac.describe_loss(inst, {"a_supported": True, "b_supported": True}, False, 50, 100),
        ac.describe_loss(inst, {}, True, 150, 100),
        ac.describe_loss(_inst(2, "none"), {"false_conflict": True}, False, 50, 100),
    ]
    assert "'1990', from d2" in texts[0] and "dropped" in texts[0]
    assert "'1995', from d0" in texts[1] and "changed" in texts[1]
    assert texts[2].startswith("OK") and "150 tokens" in texts[3] and "not in the documents" in texts[4]
    for t in texts:
        assert not re.search(r"\bD[1-7]\b|CCD|misinformation|temporal", t)


class FakeClient:
    """Compressor: echo a fixed text; judges: fixed JSON. Records every call."""
    mode, concurrency = "mock", 2

    from llm_client import LLMClient
    call_chat = LLMClient.call_chat  # same message -> (system, prompt) mapping as the real client

    def __init__(self, comp_text, csr=None, fc=False):
        self.comp_text, self.csr, self.fc, self.calls = comp_text, csr or {}, fc, []

    def call(self, model, system, prompt, max_tokens, temperature=0.0, cache_salt=None):
        self.calls.append((model, system, prompt))
        if "[Documents]" in prompt:
            return self.comp_text, {"cached": False}
        if "false_conflict" in system.lower() or "REFERENCE ANSWER" in prompt:
            return json.dumps({"false_conflict": self.fc, "rationale": "x"}), {}
        flags = {"a_supported": True, "a_distorted": False, "b_supported": True, "b_distorted": False,
                 "conflict_signaled": True, **self.csr}
        return json.dumps({**flags, "rationale": "x"}), {}


def test_score_instruction_rules():
    inst, dis = _inst(1), _inst(2, "none")
    kw = dict(examples=None, task_model=MODEL, judge_model=MODEL)
    ok = ac.score_instruction("Keep both.", [(inst, 0.5)], FakeClient("1990 [d2] vs 1995 [d0]"), **kw)[0]
    assert ok["score"] == 1.0 and ok["feedback"].startswith("OK") and not ok["over_budget"]
    assert set(ok) == {"id", "r", "score", "feedback", "compressed", "over_budget", "out_tokens", "budget"}
    lost = ac.score_instruction("Keep both.", [(inst, 0.5)], FakeClient("x", {"b_supported": False}), **kw)[0]
    assert lost["score"] == 0.0 and "'1995', from d0" in lost["feedback"]
    fc = ac.score_instruction("Keep both.", [(dis, 0.5)], FakeClient("x", fc=True), **kw)[0]
    assert fc["score"] == 0.0 and "not in the documents" in fc["feedback"]
    nofc = ac.score_instruction("Keep both.", [(dis, 0.5)], FakeClient("Item 2 was founded in 1990."), **kw)[0]
    assert nofc["score"] == 1.0 and nofc["feedback"].startswith("OK")
    garbage = ac.score_instruction("Keep both.", [(dis, 0.5)], FakeClient("Q Q # Q"), **kw)[0]   # 2.2-g floor
    assert garbage["score"] == 0.0 and "does not contain" in garbage["feedback"]
    long = FakeClient("word " * 2000)
    over = ac.score_instruction("Keep both.", [(inst, 0.3)], long, **kw)[0]
    assert over["score"] == 0.0 and over["over_budget"] and "Over budget" in over["feedback"]
    assert len(long.calls) == 2 and all("[Documents]" in c[2] for c in long.calls)  # 1 retry, judge skipped
    assert "rewrite it in at most" in long.calls[1][2]


@pytest.mark.parametrize("answer,text,kept", [("1990", "founded in 1990.", True),
                                              ("Leyton Orient F.C.", "for Leyton Orient F.C. in", True),
                                              ("Leyton Orient F.C.", "at leyton orient fc", False),  # lexical limit
                                              ("The Beatles", "beatles", True), ("199", "in 1990", False), ("", "x", False)])
def test_answer_kept(answer, text, kept):
    assert ac.answer_kept(answer, text) is kept


def test_score_instruction_uses_harness_prompt_and_records_examples():
    ac.reset_examples_seen()
    exs = json.loads((ac.ROOT_DIR / "prompts" / "c2_fewshot.json").read_text())
    fc = FakeClient("1990 1995")
    ac.score_instruction("  Instr.\n", [(_inst(1), 0.5)], fc, examples=exs, task_model=MODEL, judge_model=MODEL)
    ref = PromptCompressor("X", None, MODEL, None, instruction_text="Instr.", examples=exs)
    assert fc.calls[0][1:] == (ref.system_prompt, ref.build_prompt(_inst(1), ac.compute_budget(_inst(1)["total_tokens"], 0.5)))
    assert ac.examples_seen() == [ac.sha256_text(ac.expected_examples_block(True))]


# ------------------------------------------------------------------ DSPy bridge

def test_call_chat_maps_to_call_and_refuses_other_shapes(make_client):
    c = make_client("mock")
    t1, m1 = c.call_chat(MODEL, [{"role": "system", "content": "S"}, {"role": "user", "content": "P"}], 50)
    t2, m2 = c.call(MODEL, "S", "P", 50)
    assert t1 == t2 and m1["cache_key"] == m2["cache_key"]
    assert c.call_chat(MODEL, [{"role": "user", "content": "P"}], 50)[1]["cache_key"] == c.call(MODEL, "", "P", 50)[1]["cache_key"]
    assert c.call(MODEL, "S", "P", 50, cache_salt="r1")[1]["cache_key"] != m2["cache_key"]
    with pytest.raises(ValueError):
        c.call_chat(MODEL, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                            {"role": "user", "content": "c"}], 50)


def test_client_lm_routes_dspy_through_llm_client(make_client, no_network):
    import dspy
    c = make_client("mock")
    lm = da.ClientLM(c, MODEL)
    with dspy.context(lm=lm):  # plain ChatAdapter path (as MIPROv2's proposer uses it)
        out = dspy.Predict("question -> answer")(question="What is x?")
    assert out.answer.startswith("mock answer")
    prog = da.BudgetedCompressor("Compress.", "")
    ex = da.to_examples([(_inst(1), 0.3)])[0]
    with dspy.context(lm=lm):
        pred = prog(**ex.inputs())
    assert pred.out_tokens <= ex.budget and not pred.over_budget
    assert c.summary()["calls_total"] == 2 and c.summary()["by_model"][MODEL]["calls"] == 2


def test_harness_adapter_sends_the_harness_prompt(no_network):
    import dspy
    rec = FakeClient("short")
    lm = da.ClientLM(rec, MODEL)
    exs = json.loads((ac.ROOT_DIR / "prompts" / "c2_fewshot.json").read_text())
    inst = _inst(3)
    ex = da.to_examples([(inst, 0.5)])[0]
    with dspy.context(lm=lm):
        da.BudgetedCompressor("My instruction.\n", ac.expected_examples_block(True))(**ex.inputs())
    ref = PromptCompressor("X", None, MODEL, None, instruction_text="My instruction.", examples=exs)
    assert rec.calls[0][1:] == (ref.system_prompt, ref.build_prompt(inst, ex.budget))


# ------------------------------------------------------------------ optimizers (mock, end to end)

def test_gepa_mock_end_to_end(make_client, no_network, tmp_path):
    import apo_gepa
    import fewshot_check
    out = {"prompt": tmp_path / "c3g_seed0.txt", "program": tmp_path / "c3g_seed0.json", "log": tmp_path / "log.json"}
    log = apo_gepa.run_gepa("C3-G", 0, make_client("mock", concurrency=4), max_metric_calls=30,
                            split=_small_split(), out=out)
    assert log["status"] == "ok" and log["total_metric_calls"] <= 30 + 6
    assert out["prompt"].read_text().strip() and json.loads(out["program"].read_text())["predict"]["demos"] == []
    saved = json.loads(out["log"].read_text())
    assert saved["val_curve"][0]["iteration"] == 0 and saved["candidates"][0]["val_score"] >= 0
    assert saved["cost"]["mode"] == "mock" and saved["cost"]["calls_total"] > 0
    assert fewshot_check.check_log(saved)["valid"]

    g1 = apo_gepa.run_gepa("C3-G1", 0, make_client("mock"), max_metric_calls=12, split=_small_split(),
                           out={k: tmp_path / f"g1_{v.name}" for k, v in out.items()})
    assert g1["examples_sha256_seen"] == [ac.sha256_text("")] and fewshot_check.check_log(g1)["valid"]


def test_gepa_patience_stops_early(make_client, no_network, tmp_path):
    import apo_gepa
    out = {"prompt": tmp_path / "p.txt", "program": tmp_path / "p.json", "log": tmp_path / "l.json"}
    log = apo_gepa.run_gepa("C3-G1", 0, make_client("mock"), max_metric_calls=500, split=_small_split(),
                            out=out, patience=0)
    assert log["stop_reason"].startswith("no val improvement") and log["total_metric_calls"] < 500


def test_mipro_mock_end_to_end(make_client, no_network, tmp_path):
    pytest.importorskip("optuna")
    import apo_mipro
    import fewshot_check
    out = {"prompt": tmp_path / "c3m.txt", "program": tmp_path / "c3m.json", "log": tmp_path / "log.json"}
    log = apo_mipro.run_mipro(make_client("mock", concurrency=4), split=_small_split(), auto=None,
                              num_candidates=2, num_trials=1, minibatch=False, out=out)
    assert log["status"] == "ok" and log["demos"] == {"predict": 0} and out["prompt"].exists()
    assert fewshot_check.check_log(log)["valid"]


# ------------------------------------------------------------------ few-shot check, variants, freeze

def _program(path, demos):
    path.write_text(json.dumps({"predict": {"demos": demos, "signature": {}}, "metadata": {}}))
    return str(path)


def test_fewshot_check_pass_and_fail(tmp_path):
    import fewshot_check as fc
    good = ac.sha256_text(ac.expected_examples_block(True))
    (tmp_path / "final.txt").write_text("Instr.\n[Examples]\n## Example 1\n")
    base = {"condition": "C3-G", "seed": 0, "uses_fewshot": True, "examples_sha256_seen": [good],
            "program_path": _program(tmp_path / "ok.json", []), "final_instruction_path": str(tmp_path / "final.txt")}
    r = fc.check_log(base)
    assert r["valid"] and r["example_block_in_instruction"]["new_example_block"] is True
    assert not fc.check_log({**base, "examples_sha256_seen": [good, ac.sha256_text("")]})["valid"]
    assert not fc.check_log({**base, "examples_sha256_seen": None})["valid"]
    assert not fc.check_log({**base, "program_path": _program(tmp_path / "bad.json", [{"question": "q"}])})["valid"]
    assert not fc.check_log({**base, "condition": "C3-G1", "uses_fewshot": False})["valid"]
    protegi = {"condition": "C3-P", "seed": 0, "examples_seen": [good]}
    assert fc.check_log(protegi)["valid"]

    logs = []
    for name, log in (("ok", base), ("bad", {**base, "examples_sha256_seen": []})):
        logs.append(tmp_path / f"apo_log_{name}.json")
        logs[-1].write_text(json.dumps(log))
    assert fc.main(["--logs", str(logs[0]), "--out", str(tmp_path / "r1.json")]) == 0
    assert fc.main(["--logs", *map(str, logs), "--out", str(tmp_path / "r2.json")]) == 1
    assert json.loads((tmp_path / "r2.json").read_text())["all_valid"] is False


def test_variant_appends_verbatim_line(tmp_path):
    import apo_variants
    (tmp_path / "c3g_seed1.txt").write_text("Optimized instruction.\n")
    dst = apo_variants.make_variant(1, tmp_path)
    assert dst.name == "c3v_seed1.txt"
    assert dst.read_text() == "Optimized instruction.\n" + apo_variants.VERBATIM_LINE + "\n"
    with pytest.raises(FileNotFoundError):
        apo_variants.make_variant(2, tmp_path)


def test_freeze_collects_condition_files(monkeypatch):
    import run
    import seal
    paths = run.condition_prompt_files(["C0", "C2"], [0, 1])
    assert [p.name for p in paths] == ["c0_naive.txt", "c2_best.txt", "c2_fewshot.json"]
    with pytest.raises(FileNotFoundError, match="c3g_seed99"):   # a seed never optimized (seed 0 exists after GEPA)
        run.condition_prompt_files(["C3-G"], [99])
    got = {}
    monkeypatch.setattr(seal, "freeze_prompts", lambda ps: got.setdefault("paths", ps) and "FROZEN")
    run.main(["freeze", "--conditions", "C0", "C1", "C2", "--seeds", "0"])
    assert [p.name for p in got["paths"]] == ["c0_naive.txt", "c1_plain.txt", "c2_best.txt", "c2_fewshot.json"]
