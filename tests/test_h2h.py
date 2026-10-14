"""Amendment 2.2-b: output normalization N, M2', head-to-head rule, the `think` option, the gemma4 role row.
Fake transports only, no network. Fixtures and helpers come from test_local."""
import json

from pathlib import Path

import pytest

import pilot_select as ps
from budget import count_tokens
from compressors.llm import normalize_output
from configure_roles import assign_roles, check_family_rule, model_families
from llm_client import load_models_config
from test_local import COMP, LOCAL_CFG, FakeOllama, _tiny_instances

GEMMA4, PHI = "gemma4:e4b", "phi3.5"


# ---------- normalization N ----------
@pytest.mark.parametrize("raw,expected", [
    ("[d1] Founded in 1990.", "[d1] Founded in 1990."),                         # plain text (incl. [d1] tags) unchanged
    ('{"d3": "Founded in 1990.", "D4": "Founded in 1995."}', "Founded in 1990.\nFounded in 1995."),
    ('{"He attended Wroclaw University.": [1]}', "He attended Wroclaw University."),   # sentence key, list value
    ("{}", ""),                                                                 # empty object = empty output
    ('{"D0": "", "D1": ""}', ""),                                               # only empty values
    ('{"Emilio": [{"role": "Prof", "year": 2005}]}', "Emilio\nrole\nProf\nyear"),     # nested: keys and strings
    ("Founded in 1990.<tool_call>", "Founded in 1990."),                        # control token leak
    ("Text here.\n\n(Tokens: 95)", "Text here."),                               # trailing meta line
    ("Text here.\n\nReferences: Georgia State records.", "Text here."),         # trailing references block
    ('{ "abc', '{ "abc'),                                                       # unparseable JSON stays as it is
    # 2.2-g: run-on into the few-shot/input template is cut; a leading template (echo) is left for the checks
    ("CONFLICT: [d5] A vs [d4] B\n[d5] says A.\n\n## Example 4\n[Question]\nWho?", "CONFLICT: [d5] A vs [d4] B\n[d5] says A."),
    ("Bergen.\n\n## \n\nQuestion:\nWhich university?", "Bergen."),
    ("Text.\n[Documents]\n[d0] x", "Text."),
    ("[Question]\nWho?\n\n[Documents]\n[d0] x", "[Question]\nWho?\n\n[Documents]\n[d0] x"),
    ("# Heading kept\nText.", "# Heading kept\nText."),                          # a titled header is not template
])
def test_normalize_output(raw, expected):
    assert normalize_output(raw) == expected


def test_normalize_is_idempotent():
    once = normalize_output('{"d1": "A.\\nB."}')
    assert normalize_output(once) == once


# ---------- M2' ----------
@pytest.mark.parametrize("text,ok,qa,js", [
    ("[d1] Founded in 1990.", True, False, False),
    ("Question: who?\nAnswer: Bob", False, True, False),
    ("**Question:** who?", False, True, False),
    ("Context first.\nAnswer: Bob", False, True, False),
    ("The answer: yes, 1990", True, False, False),                               # mid-line 'answer:' is not QA mode
    ('{"a": 1}', False, False, True),
    ('{ "abc', False, False, True),                                             # JSON that N could not unwrap
    ("[1, 2]", False, False, True),
    ("", False, False, False),
    ("Sure, here it is", False, False, False),                                  # original M2 failures still count
])
def test_format_flags_v2(text, ok, qa, js):
    f = ps.format_flags_v2(text)
    assert (f["ok"], f["qa_mode"], f["json_left"]) == (ok, qa, js)


def test_tagged_answer_only_accepted_by_the_gate_variant():
    c2_line = "ANSWER: [d2] Wroclaw University\n[d2] says so."
    assert ps.format_flags_v2(c2_line)["qa_mode"] is True                      # 2.2-b definition (head-to-head) unchanged
    assert ps.format_flags_v2(c2_line, tagged_answer_ok=True)["ok"] is True    # 2.2-c gate accepts C2's line
    assert ps.format_flags_v2("Answer: Bob", tagged_answer_ok=True)["qa_mode"] is True
    assert ps.format_flags_v2("Question: who?\nANSWER: [d1] Bob", tagged_answer_ok=True)["qa_mode"] is True


def test_v1_flags_unchanged():
    assert "qa_mode" not in ps.format_flags("Question: x")            # the original 2.2 rule stays reproducible
    assert ps.format_flags("Question: x")["ok"] is True


# ---------- compressor applies N ----------
def test_compress_applies_normalization(local_h2h, tmp_path):
    raw = '{"d0": "Item founded in 1990. [d1] 1995."}'
    c, fake = local_h2h(FakeOllama(text=raw))
    recs = ps.run_model_condition(COMP, "C1", _tiny_instances()[:1], c, tmp_path / "ps", flags_fn=ps.format_flags_v2)
    r = recs[0]
    assert r["text"] == "Item founded in 1990. [d1] 1995." and r["raw_text"] == raw
    assert r["out_tokens"] == count_tokens(r["text"]) and r["format"]["ok"] and r["format"]["json_left"] is False
    m = ps.compute_metrics(recs, ps.format_flags_v2)
    assert m["normalized"] == 1 and m["M2"] == 1.0


def test_plain_output_has_no_raw_text(local_h2h, tmp_path):
    c, _ = local_h2h(FakeOllama(text="[d0] Item founded in 1990."))
    r = ps.run_model_condition(COMP, "C1", _tiny_instances()[:1], c, tmp_path / "ps", flags_fn=ps.format_flags_v2)[0]
    assert r["raw_text"] is None


# ---------- think option ----------
def _think_cfg():
    return {**LOCAL_CFG, "models": {**LOCAL_CFG["models"], GEMMA4: {"provider": "ollama", "num_ctx": 8192, "think": False}}}


class ThinkingFake(FakeOllama):
    def __call__(self, path, payload):
        out = super().__call__(path, payload)
        if path == "/api/chat":
            out["message"]["thinking"] = "x" * 7
        return out


def test_think_sent_only_for_configured_models_and_in_cache_key(make_client):
    fake = ThinkingFake()
    c = make_client("execute", models_config=_think_cfg(), ollama_transport_fn=fake, max_cost_usd=0.0)
    _, meta = c.call(GEMMA4, "s", "p", max_tokens=10)
    c.call(PHI, "s", "p", max_tokens=10)
    g, p = fake.chats
    assert g["think"] is False and "think" not in p
    assert meta["ollama"]["think"] is False and meta["ollama"]["thinking_chars"] == 7
    assert c.provider_opts(GEMMA4) == {"num_ctx": 8192, "json_mode": False, "seed": 0, "think": False}
    assert c.provider_opts(PHI) == {"num_ctx": 8192, "json_mode": False, "seed": 0}       # existing cache keys unchanged
    cfg_on = _think_cfg()
    cfg_on["models"][GEMMA4] = {"provider": "ollama", "num_ctx": 8192, "think": True}
    c2 = make_client("execute", models_config=cfg_on, ollama_transport_fn=FakeOllama(), max_cost_usd=0.0)
    assert c._key(GEMMA4, "s", "p", 0.0, 10) != c2._key(GEMMA4, "s", "p", 0.0, 10)       # think is part of the key


class RepeatAbortFake(FakeOllama):
    """Ollama >= 0.40: HTTP 500 'prediction aborted, token repeat limit reached' (amendment 2.2-f)."""
    def __init__(self, abort_model, **kw):
        super().__init__(**kw)
        self.abort_model, self.aborts = abort_model, 0

    def __call__(self, path, payload):
        if path == "/api/chat" and payload["model"] == self.abort_model:
            self.aborts += 1
            from llm_client import RetryableHTTPError
            raise RetryableHTTPError('Ollama HTTP 500: {"error":"prediction aborted, token repeat limit reached"}')
        return super().__call__(path, payload)


def test_repeat_abort_is_an_empty_output_not_retried(make_client):
    fake = RepeatAbortFake(PHI)
    c = make_client("execute", models_config=_think_cfg(), ollama_transport_fn=fake, max_cost_usd=0.0)
    text, meta = c.call(PHI, "s", "p", max_tokens=10)
    assert text == "" and meta["ollama"]["done_reason"] == "repeat_limit" and fake.aborts == 1   # no retry
    assert {"model": PHI, "keep_alive": 0} in fake.chats                    # 2.2-h: the degraded runner is unloaded
    text2, meta2 = c.call(PHI, "s", "p", max_tokens=10)                                        # cached, deterministic
    assert text2 == "" and meta2["cached"] and fake.aborts == 1
    text3, _ = c.call(GEMMA4, "s", "p", max_tokens=10)                                         # other models unaffected
    assert text3 == "compressed text"


def test_cache_dir_from_env(monkeypatch, tmp_path):
    from llm_client import LLMClient
    monkeypatch.setenv("CCD_CACHE_DIR", str(tmp_path / "rerun"))
    assert LLMClient(mode="mock").cache_dir == tmp_path / "rerun"
    assert LLMClient(mode="mock", cache_dir=tmp_path / "own").cache_dir == tmp_path / "own"   # explicit wins


def test_other_5xx_still_retried(make_client):
    fake = FakeOllama(fail_first=2)
    c = make_client("execute", models_config=_think_cfg(), ollama_transport_fn=fake, max_cost_usd=0.0)
    assert c.call(PHI, "s", "p", max_tokens=10)[0] == "compressed text"


# ---------- head-to-head rule ----------
def _hr(bad_c0=0, bad_c1=0, short=0, seconds=10.0, n=100):
    """n outputs per condition; `bad_*` preamble failures, `short` outputs below 10% of the budget (split evenly)."""
    recs = []
    for c, bad in (("C0", bad_c0), ("C1", bad_c1)):
        for i in range(n):
            recs.append({"id": f"{c}{i}", "condition": c, "budget": 100, "out_tokens": 5 if i < short // 2 else 50,
                         "text": "Sure, here" if i < bad else "[d1] text", "retried": False, "seconds": seconds})
    return recs


def _sel(**by_model):
    pooled = {m: ps.compute_metrics(r, ps.format_flags_v2) for m, r in by_model.items()}
    by_cond = {m: {c: ps.compute_metrics([x for x in r if x["condition"] == c], ps.format_flags_v2)
                   for c in ("C0", "C1")} for m, r in by_model.items()}
    return ps.select_h2h(pooled, by_cond)


def test_h2h_single_gate_pass():
    s = _sel(**{PHI: _hr(bad_c0=1, bad_c1=19), GEMMA4: _hr()})                     # phi M2' 0.90 and G 0.18 -> out
    assert (s["selected"], s["step"], s["passed"]) == (GEMMA4, "gate", [GEMMA4])


def test_h2h_gap_gate_alone_fails():
    s = _sel(**{PHI: _hr(bad_c0=0, bad_c1=6), GEMMA4: _hr(bad_c0=3, bad_c1=3)})    # phi M2' 0.97 but G 0.06 -> out
    assert s["selected"] == GEMMA4 and s["gate"][PHI] is False and s["gate"][GEMMA4] is True


def test_h2h_gap_boundary_is_inclusive():
    s = _sel(**{PHI: _hr(bad_c0=0, bad_c1=5), GEMMA4: _hr(bad_c0=40, bad_c1=40)})   # G = 0.05 exactly passes
    assert s["gate"][PHI] is True and s["selected"] == PHI


def test_h2h_m3_decides_when_gap_over_010():
    s = _sel(**{PHI: _hr(short=0), GEMMA4: _hr(short=40)})                          # M3 1.00 vs 0.80
    assert (s["selected"], s["step"]) == (PHI, "M3")
    s = _sel(**{PHI: _hr(short=40), GEMMA4: _hr(short=0)})
    assert (s["selected"], s["step"]) == (GEMMA4, "M3")


def test_h2h_m3_within_010_then_smaller_gap():
    s = _sel(**{PHI: _hr(bad_c0=0, bad_c1=4, short=20), GEMMA4: _hr(bad_c0=2, bad_c1=2, short=0)})  # M3 0.90 vs 1.00
    assert (s["selected"], s["step"]) == (GEMMA4, "G")                              # M3 diff 0.10 = tie, G 0.00 < 0.04


def test_h2h_speed_then_incumbent():
    s = _sel(**{PHI: _hr(seconds=9.0), GEMMA4: _hr(seconds=20.0)})
    assert (s["selected"], s["step"]) == (PHI, "speed")
    s = _sel(**{PHI: _hr(seconds=30.0), GEMMA4: _hr(seconds=12.0)})
    assert s["selected"] == GEMMA4
    s = _sel(**{GEMMA4: _hr(), PHI: _hr()})                                         # dict order does not matter
    assert s["selected"] == PHI                                                     # exact tie -> incumbent (list order)


def test_h2h_none_pass_picks_higher_m2_and_flags():
    s = _sel(**{PHI: _hr(bad_c0=10, bad_c1=10), GEMMA4: _hr(bad_c0=20, bad_c1=20)})
    assert (s["selected"], s["step"], s["passed"]) == (PHI, "none", [])
    assert s["notes"]
    s = _sel(**{PHI: _hr(bad_c0=20, bad_c1=20), GEMMA4: _hr(bad_c0=10, bad_c1=10)})
    assert s["selected"] == GEMMA4


# ---------- roles and repo config ----------
def test_gemma4_role_row_family_rule_on_real_config():
    roles = assign_roles(GEMMA4)
    assert roles["compressor"] == GEMMA4 and roles["downstream"] == "same_as_compressor"
    assert roles["opt_judge"] == "llama3.1:8b" and roles["reflection"] == "mistral-small"
    assert check_family_rule(roles, model_families()) == []


def test_repo_config_has_gemma4_with_think_false():
    cfg = load_models_config()
    assert cfg["models"][GEMMA4] == {"provider": "ollama", "family": "google", "num_ctx": 8192, "think": False}
    assert all("think" not in cfg["models"][m] for m in ps.CANDIDATES)             # original candidates untouched


# ---------- end to end (mock) ----------
@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data" / "processed" / "dev.jsonl").exists(),
                    reason="dev data not built (see README: Rebuilding the data)")
def test_h2h_mock_run_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "H2H_MOCK_DIR", tmp_path / "mock")
    ps.main(["--h2h", "--mode", "mock", "--limit", "2"])
    md = (tmp_path / "mock" / ps.H2H_REPORT.name).read_text()
    assert "압축기 1:1 비교 결과" in md and "게이트(M1" in md and "선정 결과" in md
    assert len(list((tmp_path / "mock").glob("*_C*.jsonl"))) == 4                   # 2 models x 2 conditions


def test_h2h_plan_runs_nothing(capsys, monkeypatch):
    monkeypatch.setattr(ps, "LLMClient", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no client")))
    ps.main(["--h2h"])
    assert "400 compressions" in capsys.readouterr().out                          # 2 models x 2 conditions x 100


def test_h2h_limit_is_capped():
    with pytest.raises(SystemExit):
        ps.main(["--h2h", "--limit", "101"])


@pytest.fixture
def local_h2h(make_client):
    def make(fake=None):
        fake = fake or FakeOllama()
        return make_client("execute", models_config=LOCAL_CFG, ollama_transport_fn=fake, max_cost_usd=0.0), fake
    return make
