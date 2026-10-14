"""Local Ollama provider, role assignment and the compressor selection pilot. Fake transports only, no network."""
import json
import shutil

import pytest

import llm_client
import pilot_select as ps
from configure_roles import ROLE_TABLE, SPLASH, assign_roles, write_roles
from conftest import CFG, PRICED
from llm_client import BudgetExceeded, RetryableHTTPError, cache_key, load_models_config, resolve_roles
from run import run_condition
from test_e2e_mock import _inst

COMP, JUDGE = "qwen2.5:7b", "gemma2:9b"
LOCAL_CFG = {
    **CFG,
    "roles": {"compressor": COMP, "opt_judge": JUDGE, "eval_judge": "mistral-small",
              "downstream": "same_as_compressor", "reflection": "gemma2:27b", "aux_compressor": None},
    "role_options": {"opt_judge": {"json_mode": True}, "eval_judge": {"json_mode": True}},
    "ollama": {"seed": 0, "num_ctx": 8192, "concurrency": 1},
    "models": {m: {"provider": "ollama", "num_ctx": 8192} for m in ROLE_TABLE | {"gemma2:27b": 0, "mistral-small": 0}},
}


class FakeOllama:
    def __init__(self, text="compressed text", fail_first=0):
        self.chats, self.tags_calls, self.fail_first, self.text = [], 0, fail_first, text

    def __call__(self, path, payload):
        if path == "/api/tags":
            self.tags_calls += 1
            return {"models": [{"name": f"{m}:latest" if ":" not in m else m, "digest": f"sha-{m}",
                                "size": 1, "modified_at": "t"} for m in LOCAL_CFG["models"]]}
        if self.fail_first:
            self.fail_first -= 1
            raise RetryableHTTPError("connection refused")
        self.chats.append(payload)
        return {"message": {"role": "assistant", "content": self.text}, "done_reason": "stop",
                "prompt_eval_count": 1234, "eval_count": 56, "total_duration": 2_500_000_000}


@pytest.fixture
def local(make_client):
    def make(mode="execute", fake=None, **kw):
        fake = fake or FakeOllama()
        kw.setdefault("max_cost_usd", 0.0 if mode == "execute" else None)
        return make_client(mode, models_config=LOCAL_CFG, ollama_transport_fn=fake, **kw), fake
    return make


def test_request_body_and_meta(local, tmp_path):
    c, fake = local()
    text, meta = c.call(COMP, "SYS", "PROMPT", max_tokens=300, temperature=0.0)
    c.call(JUDGE, "SYS", "PROMPT", max_tokens=200)
    comp_req, judge_req = fake.chats
    assert comp_req["stream"] is False and "format" not in comp_req
    assert comp_req["options"] == {"num_ctx": 8192, "seed": 0, "temperature": 0.0, "num_predict": 300}
    assert comp_req["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "PROMPT"}]
    assert judge_req["format"] == "json" and judge_req["options"]["num_ctx"] == 8192
    assert meta["model_digest"] == f"sha-{COMP}" and meta["provider"] == "ollama"
    assert meta["ollama"]["prompt_eval_count"] == 1234 and meta["ollama"]["total_duration_ns"] == 2_500_000_000
    assert fake.tags_calls == 1
    man = json.loads((tmp_path / "models_manifest.json").read_text())
    assert man[COMP]["digest"] == f"sha-{COMP}" and man[JUDGE]["digest"] == f"sha-{JUDGE}"


def test_cache_keys_separate_providers_and_options(local):
    c, fake = local()
    plain = cache_key(COMP, "s", "p", 0.0, 10)
    assert c._key(COMP, "s", "p", 0.0, 10) != plain                       # ollama options in the key
    assert c._key(COMP, "s", "p", 0.0, 10) != c._key(JUDGE, "s", "p", 0.0, 10)
    assert c.provider_opts(JUDGE)["json_mode"] and not c.provider_opts(COMP)["json_mode"]
    assert c._key(PRICED, "s", "p", 0.0, 10) == cache_key(PRICED, "s", "p", 0.0, 10)  # OpenRouter key unchanged
    c.call(COMP, "s", "p", max_tokens=10)
    text, meta = c.call(COMP, "s", "p", max_tokens=10)
    assert len(fake.chats) == 1 and meta["cached"] and meta["ollama"]["prompt_eval_count"] == 1234


def test_json_off_when_model_also_has_plain_role(make_client):
    cfg = {**LOCAL_CFG, "roles": {**LOCAL_CFG["roles"], "reflection": JUDGE}}
    c = make_client("dry_run", models_config=cfg, ollama_transport_fn=FakeOllama())
    assert not c.provider_opts(JUDGE)["json_mode"]


def test_price_zero_and_max_cost_zero(local, monkeypatch):
    c, fake = local()
    for i in range(3):
        c.call(COMP, "s", f"p{i}", max_tokens=10)
    s = c.summary()
    assert c.price(COMP) == (0.0, 0.0) and s["usd_spent"] == 0 and s["usd_est_total"] == 0 and len(fake.chats) == 3
    # OpenRouter under the same client is still refused: cap 0 -> no request is sent
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-value")
    with pytest.raises(BudgetExceeded):
        c.call(PRICED, "s", "p", max_tokens=10)
    with pytest.raises(RuntimeError, match="TO_CONFIRM"):
        c.call("test/unpriced", "s", "p", max_tokens=10)


def test_openrouter_execute_still_needs_key(local, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "ROOT_DIR", tmp_path)
    c, _ = local(max_cost_usd=5.0)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        c.call(PRICED, "s", "p", max_tokens=10)
    c.call(COMP, "s", "p", max_tokens=10)  # local call needs no key


def test_connection_retries(local):
    c, fake = local(fake=FakeOllama(fail_first=3))
    assert c.call(COMP, "s", "p", max_tokens=10)[0] == "compressed text"
    c2, _ = local(fake=FakeOllama(fail_first=4))
    with pytest.raises(RetryableHTTPError):
        c2.call(COMP, "s", "p2", max_tokens=10)


def test_missing_ollama_model(make_client):
    cfg = {**LOCAL_CFG, "models": {**LOCAL_CFG["models"], "llama3.2:1b": {"provider": "ollama"}}}
    c = make_client("execute", max_cost_usd=0.0, models_config=cfg, ollama_transport_fn=FakeOllama())
    with pytest.raises(RuntimeError, match="ollama pull"):
        c.call("llama3.2:1b", "s", "p", max_tokens=10)


def test_mock_and_dry_run_never_touch_ollama(make_client):
    def boom(path, payload):
        raise AssertionError("no network")
    for mode in ("mock", "dry_run"):
        c = make_client(mode, models_config=LOCAL_CFG, ollama_transport_fn=boom)
        _, meta = c.call(COMP, "Compress.", "[Documents]\nabc\n[Budget]\nat most 5 tokens", max_tokens=10)
        assert meta["model_digest"] is None


# ---------- roles ----------
def test_assign_roles_table():
    # revised 2.3 table (2026-10-09): details in tests/test_splash.py
    assert assign_roles("llama3.1:8b")["eval_judge"] == SPLASH and assign_roles("qwen2.5:7b")["reflection"] == SPLASH
    with pytest.raises(ValueError):
        assign_roles("gemma2:27b")
    with pytest.raises(ValueError):
        assign_roles(SPLASH)                                      # not a compressor candidate (2.1)


def test_write_roles_keeps_rest_of_file(tmp_path):
    p = tmp_path / "models.yaml"
    shutil.copy(llm_client.MODELS_CONFIG, p)
    before = p.read_text()
    backup = write_roles(assign_roles("gemma2:9b"), p)
    assert backup.read_text() == before
    cfg = load_models_config(p)
    assert cfg["roles"]["compressor"] == "gemma2:9b" and cfg["roles"]["downstream"] == "gemma2:9b"
    assert cfg["roles"]["aux_compressor"] is None and cfg["models"] and cfg["ollama"]["num_ctx"] == 8192
    after = p.read_text()
    assert after.split("roles:")[0] == before.split("roles:")[0]
    assert after.split("role_options:")[1] == before.split("role_options:")[1]


def test_resolve_roles_and_repo_config():
    assert resolve_roles({"compressor": "a", "downstream": "same_as_compressor"})["downstream"] == "a"
    cfg = load_models_config()
    assert cfg["roles"]["downstream"] == cfg["roles"]["compressor"]
    assert all(cfg["models"][m]["provider"] == "ollama" and cfg["models"][m]["num_ctx"] == 8192
               for m in ps.CANDIDATES + ["gemma2:27b", "mistral-small"])


# ---------- pilot_select ----------
@pytest.mark.parametrize("text,ok", [
    ("[d1] Founded in 1990.", True), ("", False), ("   ", False), ("Here is the compressed text: x", False),
    ("**Sure!** x", False), ("Certainly, x", False), ("I'm sorry, but I cannot", False),
    ("Could you clarify the question?", False), ("Surely the dates differ", True), ("Here, documents say", True),
    ("I can’t do that", False),
])
def test_format_flags(text, ok):
    assert ps.format_flags(text)["ok"] is ok


def _recs(n_over=0, n_bad=0, n_short=0, seconds=10.0, n=100):
    recs = []
    for i in range(n):
        out = 200 if i < n_over else (5 if i < n_over + n_short else 50)
        recs.append({"id": f"x{i}", "condition": "C0" if i % 2 else "C1", "budget": 100, "out_tokens": out,
                     "text": "Sure, here" if n_over + n_short <= i < n_over + n_short + n_bad else "[d1] text",
                     "retried": False, "seconds": seconds})
    return recs


def test_metrics():
    m = ps.compute_metrics(_recs(n_over=3, n_bad=2, n_short=4, seconds=7.0))
    assert (m["M1"], m["M2"], m["M3"], m["median_s"]) == (0.97, 0.98, 0.96, 7.0)
    assert ps.compute_metrics([{"out_tokens": 10, "budget": 100, "text": "x", "retried": False, "seconds": 1}])["M3"] == 1.0


def test_select_normal():
    met = {"gemma2:9b": ps.compute_metrics(_recs(n_short=10)),            # M3 0.90
           "llama3.1:8b": ps.compute_metrics(_recs()),                    # M3 1.00
           "phi3.5": ps.compute_metrics(_recs(n_bad=6)),                  # M2 0.94 -> out
           "qwen2.5:7b": ps.compute_metrics(_recs(n_over=6))}             # M1 0.94 -> out
    sel = ps.select_compressor(met)
    assert sel["rule"] == 2 and sel["selected"] == "llama3.1:8b" and sel["eligible"] == ["gemma2:9b", "llama3.1:8b"]


def test_select_tie_within_002_prefers_faster():
    met = {"llama3.1:8b": ps.compute_metrics(_recs(seconds=20.0)),           # M3 1.00
           "qwen2.5:7b": ps.compute_metrics(_recs(n_short=2, seconds=12.0)),  # M3 0.98 (diff 0.02 -> tie)
           "qwen2.5:14b": ps.compute_metrics(_recs(n_short=3, seconds=5.0))}  # M3 0.97 (diff 0.03 -> out)
    sel = ps.select_compressor(met)
    assert sel["tie_set"] == ["llama3.1:8b", "qwen2.5:7b"] and sel["selected"] == "qwen2.5:7b"
    # float edge: 0.97 - 0.95 is 0.020000000000000018 in binary floating point; still a tie
    met = {"llama3.1:8b": ps.compute_metrics(_recs(n_short=3, seconds=20.0)),
           "phi3.5": ps.compute_metrics(_recs(n_short=5, seconds=9.0))}
    assert ps.select_compressor(met)["selected"] == "phi3.5"


def test_select_all_fail_rule3():
    met = {"gemma2:9b": ps.compute_metrics(_recs(n_over=10, n_bad=0)),   # (0.90+1.00)/2 = 0.95
           "phi3.5": ps.compute_metrics(_recs(n_over=2, n_bad=10)),      # (0.98+0.90)/2 = 0.94
           "qwen2.5:7b": ps.compute_metrics(_recs(n_over=20))}           # 0.90
    sel = ps.select_compressor(met)
    assert sel["rule"] == 3 and sel["selected"] == "gemma2:9b" and sel["eligible"] == []
    assert any("규칙 3" in n for n in sel["notes"])


def _tiny_instances():
    return [_inst(1, "misinformation", "1990", "1995"), _inst(2, "temporal", "2001", "2011"),
            _inst(3, "misinformation", "1970", "1975")]


def test_pilot_execute_with_fake_ollama_and_resume(local, tmp_path):
    fake = FakeOllama(text="[d0] Item founded in 1990. [d1] 1995.")
    c, _ = local(fake=fake)
    insts = _tiny_instances()
    path = ps.out_file(tmp_path / "ps", COMP, "C1")
    path.parent.mkdir()
    first = ps.run_model_condition(COMP, "C1", insts[:1], c, tmp_path / "ps")
    path.write_text(path.read_text() + '{"id": "t2", "trunc')                # interrupted mid-line
    recs = ps.run_model_condition(COMP, "C1", insts, c, tmp_path / "ps")
    assert [r["id"] for r in recs] == ["t1", "t2", "t3"] and recs[0] == first[0]
    assert len(fake.chats) == 3                                              # t1 not re-run
    assert len(path.read_text().splitlines()) == 3
    assert path.name == "qwen2.5-7b_C1.jsonl"
    r = recs[1]
    assert r["model_digest"] == f"sha-{COMP}" and r["seconds"] == 2.5 and r["format"]["ok"]
    assert r["ollama"][0]["num_ctx"] == 8192 and r["ollama"][0]["prompt_eval_count"] == 1234
    res = ps.write_report(tmp_path / "rep.md", {COMP: recs}, n_instances=3, mode="execute", complete=False,
                          ollama_cfg=c.ollama)
    md = (tmp_path / "rep.md").read_text()
    assert "부분 실행" in md and '"here is"' in md and '"i\'m sorry"' in md and res["selection"]["selected"] == COMP


def test_pilot_plan_runs_nothing(capsys, monkeypatch):
    monkeypatch.setattr(ps, "LLMClient", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no client")))
    ps.main([])
    out = capsys.readouterr().out
    assert "500 compressions" in out and "1.4-3.1 h" in out


# ---------- run.py ----------
def test_run_resumes_from_partial(tmp_path, make_client):
    data = tmp_path / "mini.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in _tiny_instances()))
    out = tmp_path / "out"
    c0 = make_client("mock")
    full = run_condition("C0", "dev", 0.3, c0, results_dir=out, data_path=data)
    final = out / "C0_dev_r0.3_s0.jsonl"
    lines = final.read_text().splitlines()
    final.unlink()
    (out / "C0_dev_r0.3_s0.jsonl.partial").write_text(lines[0] + "\n" + lines[1][:20])
    c = make_client("mock")
    recs = run_condition("C0", "dev", 0.3, c, results_dir=out, data_path=data)
    assert sorted(r["id"] for r in recs) == sorted(r["id"] for r in full)
    assert c.summary()["calls_total"] < c0.summary()["calls_total"]   # the kept instance is not re-run
    assert len(final.read_text().splitlines()) == 3 and "model_digests" in recs[-1]
