"""Splash provider (incoai/Qwen3.8-27B-Splash), revised 2.3 role table, 2.4 fallbacks, family rule.
Fake transports and temp files only: no Splash/Ollama server is contacted."""
import json

import pytest

from configure_roles import (FALLBACK_ORDER, ROLE_TABLE, SPLASH, assign_roles, check_family_rule, fallback,
                             model_families)
from conftest import CFG
from llm_client import cache_key, load_models_config, resolve_roles

BASE = "http://127.0.0.1:8010/v1"
FAM = model_families()
SPLASH_CFG = {
    **CFG,
    "roles": {"compressor": "qwen2.5:7b", "opt_judge": "gemma2:9b", "eval_judge": SPLASH,
              "downstream": "same_as_compressor", "reflection": "mistral-small", "aux_compressor": None},
    "role_options": {"opt_judge": {"json_mode": True}, "eval_judge": {"json_mode": True}},
    "models": {SPLASH: {"provider": "splash", "family": "alibaba", "base_url": BASE},
               "qwen2.5:7b": {"provider": "ollama"}},
}
USAGE = {"prompt_tokens": 907, "completion_tokens": 40, "total_tokens": 947,
         "prompt_tokens_details": {"cached_tokens": 512}, "completion_tokens_details": {"reasoning_tokens": 0}}


class FakeSplash:
    def __init__(self, ids=(SPLASH,)):
        self.posts, self.gets, self.ids = [], [], ids

    def __call__(self, url, payload):
        if payload is None:
            self.gets.append(url)
            return {"object": "list", "data": [{"id": i, "object": "model"} for i in self.ids]}
        self.posts.append((url, payload))
        return {"choices": [{"message": {"role": "assistant", "content": '{"a": 1}'}, "finish_reason": "stop"}],
                "usage": USAGE}


def _manifest(tmp_path, n=1):
    for i in range(n):
        d = tmp_path / "snapshots" / f"rev{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"artifact_set_sha256": f"abc{i}", "model": "Qwen3.8-27B"}))
    return str(tmp_path / "snapshots" / "*" / "manifest.json")


@pytest.fixture
def splash(make_client, tmp_path):
    def make(mode="execute", cfg=SPLASH_CFG, fake=None, **kw):
        fake = fake or FakeSplash()
        kw.setdefault("max_cost_usd", 0.0 if mode == "execute" else None)
        cfg = {**cfg, "models": {**cfg["models"], SPLASH: {**cfg["models"][SPLASH],
                                                            "manifest_path": _manifest(tmp_path / "hf")}}}
        return make_client(mode, models_config=cfg, splash_transport_fn=fake,
                           ollama_transport_fn=lambda p, b: (_ for _ in ()).throw(AssertionError("ollama")), **kw), fake
    return make


def test_request_body_usage_and_manifest(splash, tmp_path):
    c, fake = splash()
    text, meta = c.call(SPLASH, "SYS", "PROMPT", max_tokens=200, temperature=0.0)
    (url, body), = fake.posts
    assert url == f"{BASE}/chat/completions" and text == '{"a": 1}'
    assert body == {"model": SPLASH, "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "PROMPT"}],
                    "temperature": 0.0, "max_tokens": 200, "seed": 0, "reasoning_effort": "none",
                    "response_format": {"type": "json_object"}}            # eval_judge -> JSON mode
    assert meta["provider"] == "splash" and meta["model_digest"] == "abc0" and meta["cost_usd"] == 0.0
    assert {k: meta["splash"][k] for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens")} \
        == {"prompt_tokens": 907, "completion_tokens": 40, "cached_tokens": 512, "reasoning_tokens": 0}
    assert fake.gets == [f"{BASE}/models"]
    man = json.loads((tmp_path / "models_manifest.json").read_text())[SPLASH]
    assert man["artifact_set_sha256"] == man["digest"] == "abc0" and man["served_model_ids"] == [SPLASH]
    _, meta2 = c.call(SPLASH, "SYS", "PROMPT", max_tokens=200)             # cache hit keeps the usage record
    assert meta2["cached"] and meta2["splash"]["cached_tokens"] == 512 and len(fake.posts) == 1
    assert len(fake.gets) == 1


def test_non_judge_role_and_reasoning_field_off(splash):
    cfg = {**SPLASH_CFG, "roles": {**SPLASH_CFG["roles"], "eval_judge": "mistral-small", "reflection": SPLASH}}
    cfg["models"] = {**cfg["models"], SPLASH: {**cfg["models"][SPLASH], "reasoning_effort": None}}
    c, fake = splash(cfg=cfg)
    c.call(SPLASH, "s", "p", max_tokens=10)
    body = fake.posts[0][1]
    assert "response_format" not in body and "reasoning_effort" not in body


def test_cache_key_and_price(splash):
    c, _ = splash(mode="dry_run")
    k = c._key(SPLASH, "s", "p", 0.0, 10)
    assert k != cache_key(SPLASH, "s", "p", 0.0, 10)                        # provider options in the key
    assert k != c._key("qwen2.5:7b", "s", "p", 0.0, 10)
    cfg = {**SPLASH_CFG, "roles": {**SPLASH_CFG["roles"], "eval_judge": "mistral-small", "reflection": SPLASH}}
    assert splash(mode="dry_run", cfg=cfg)[0]._key(SPLASH, "s", "p", 0.0, 10) != k   # json_mode differs
    assert c.price(SPLASH) == (0.0, 0.0)
    _, meta = c.call(SPLASH, "s", "p", max_tokens=10)
    assert meta["dry_run"] and meta["cost_usd_est"] == 0.0 and meta["model_digest"] is None


def test_max_cost_zero_and_no_server_check_outside_execute(splash):
    c, fake = splash()
    for i in range(3):
        c.call(SPLASH, "s", f"p{i}", max_tokens=10)
    assert c.summary()["usd_spent"] == 0 and len(fake.posts) == 3
    m, mfake = splash(mode="mock")
    m.call(SPLASH, "Return JSON only.", "x", max_tokens=10)
    assert not mfake.gets and not mfake.posts


def test_version_errors(splash, make_client, tmp_path):
    c, _ = splash(fake=FakeSplash(ids=("other/model",)))
    with pytest.raises(RuntimeError, match="does not serve"):
        c.call(SPLASH, "s", "p", max_tokens=10)
    cfg = {**SPLASH_CFG, "models": {**SPLASH_CFG["models"],
                                    SPLASH: {**SPLASH_CFG["models"][SPLASH], "manifest_path": _manifest(tmp_path / "two", 2)}}}
    c2 = make_client("execute", max_cost_usd=0.0, models_config=cfg, splash_transport_fn=FakeSplash())
    with pytest.raises(RuntimeError, match="matched 2 files"):
        c2.call(SPLASH, "s", "p", max_tokens=10)


def test_repo_config_has_splash_and_families():
    cfg = load_models_config()
    m = cfg["models"][SPLASH]
    assert m["provider"] == "splash" and m["family"] == "alibaba" and m["base_url"] == BASE
    assert m["manifest_path"].endswith("models--incoai--Qwen3.8-27B-Splash/snapshots/*/manifest.json")
    assert {k: FAM[k] for k in ("llama3.1:8b", "gemma2:9b", "gemma2:27b", "qwen2.5:7b", "qwen2.5:14b", "phi3.5",
                                "mistral-small")} == {
        "llama3.1:8b": "meta", "gemma2:9b": "google", "gemma2:27b": "google", "qwen2.5:7b": "alibaba",
        "qwen2.5:14b": "alibaba", "phi3.5": "microsoft", "mistral-small": "mistral"}


# ---------- roles (preregistration 2.3 revised table, 2.4) ----------
def test_assign_roles_revised_table():
    for comp in ("llama3.1:8b", "phi3.5"):
        assert assign_roles(comp) == {"compressor": comp, "opt_judge": "gemma2:9b", "eval_judge": SPLASH,
                                      "downstream": "same_as_compressor", "reflection": "mistral-small",
                                      "aux_compressor": None}
    for comp in ("qwen2.5:7b", "qwen2.5:14b"):
        r = assign_roles(comp)
        assert (r["opt_judge"], r["reflection"], r["eval_judge"]) == ("gemma2:9b", SPLASH, "mistral-small")
    r = assign_roles("gemma2:9b")
    assert (r["opt_judge"], r["reflection"], r["eval_judge"]) == ("llama3.1:8b", "mistral-small", SPLASH)
    for comp in ROLE_TABLE:
        assert check_family_rule(assign_roles(comp)) == []


def _json_conflicts(roles):
    roles = resolve_roles(roles)
    judge = {roles["opt_judge"], roles["eval_judge"]}
    plain = {m for k, m in roles.items() if k not in ("opt_judge", "eval_judge") and m}
    return judge & plain


def test_no_json_conflict_in_table_and_only_documented_one_in_fallbacks():
    assert all(not _json_conflicts(assign_roles(c)) for c in ROLE_TABLE)
    found = set()
    for comp in ROLE_TABLE:
        for role, order in FALLBACK_ORDER.items():
            for m in order:
                roles = {**assign_roles(comp), role: m}
                if not check_family_rule(roles) and _json_conflicts(roles):
                    found.add((comp, role, m))
    assert found == {("qwen2.5:14b", "opt_judge", "qwen2.5:14b")}           # documented in configure_roles


def test_fallback_order():
    assert fallback("eval_judge", "llama3.1:8b", set()) == SPLASH
    assert fallback("eval_judge", "llama3.1:8b", {SPLASH}) == "qwen2.5:14b"
    # gemma2:27b (google = opt judge gemma2:9b) and mistral-small (= reflection) break the family rule
    assert fallback("eval_judge", "llama3.1:8b", {SPLASH, "qwen2.5:14b"}) is None
    assert fallback("eval_judge", "gemma2:9b", {SPLASH}) == "qwen2.5:14b"
    # qwen compressors: qwen2.5:14b (alibaba) and gemma2:27b (google = opt judge) are excluded -> nothing left
    assert fallback("eval_judge", "qwen2.5:7b", {"mistral-small"}) is None
    assert fallback("opt_judge", "qwen2.5:7b", {"gemma2:9b"}) == "qwen2.5:14b"
    assert fallback("opt_judge", "qwen2.5:7b", {"gemma2:9b", "qwen2.5:14b"}) == "gemma2:27b"
    assert fallback("opt_judge", "qwen2.5:7b", {"gemma2:9b", "qwen2.5:14b", "gemma2:27b"}) is None
    # llama row: qwen2.5:14b as opt judge would share alibaba with the Splash eval judge
    assert fallback("opt_judge", "llama3.1:8b", {"gemma2:9b"}) == "gemma2:27b"
    # second replacement sees the first: eval judge moved to qwen2.5:14b, opt judge fallback must avoid alibaba
    roles = {**assign_roles("gemma2:9b"), "eval_judge": "qwen2.5:14b"}
    assert fallback("opt_judge", "gemma2:9b", {"llama3.1:8b"}, roles=roles) == "gemma2:27b"


def test_check_family_rule_violations():
    base = assign_roles("llama3.1:8b")
    for role, model in (("compressor", "qwen2.5:7b"), ("opt_judge", "qwen2.5:14b"), ("reflection", SPLASH)):
        bad = check_family_rule({**base, role: model})
        assert len(bad) == 1 and role in bad[0] and "alibaba" in bad[0]
    assert "no family" in check_family_rule({**base, "reflection": "unknown/model"})[0]
    # opt judge and instruction generator may share a family
    assert check_family_rule({**base, "reflection": "gemma2:27b"}) == []



def test_cli_checks_family_rule_on_written_file(tmp_path, monkeypatch, capsys):
    import shutil

    import configure_roles
    p = tmp_path / "models.yaml"
    shutil.copy(configure_roles.MODELS_CONFIG, p)
    monkeypatch.setattr(configure_roles, "MODELS_CONFIG", p)
    configure_roles.main(["--compressor", "qwen2.5:7b"])
    assert "family rule (2.3): ok" in capsys.readouterr().out
    assert load_models_config(p)["roles"]["reflection"] == SPLASH
    monkeypatch.setattr(configure_roles, "ROLE_TABLE", {**ROLE_TABLE, "qwen2.5:7b": ("gemma2:9b", SPLASH, "qwen2.5:14b")})
    with pytest.raises(SystemExit, match="family rule"):
        configure_roles.main(["--compressor", "qwen2.5:7b"])
