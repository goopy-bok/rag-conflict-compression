import hashlib
import io
import json
import sqlite3
import urllib.error

import pytest

import llm_client
from budget import count_tokens
from compressors import PromptCompressor, retry_note
from conftest import PRICED
from llm_client import BudgetExceeded, RetryableHTTPError, cache_key, legacy_cache_key
from mock_llm import respond

ROOT = llm_client.ROOT_DIR


def _ok(text="ok", cost=None, pt=10, ct=5):
    usage = {"prompt_tokens": pt, "completion_tokens": ct}
    if cost is not None:
        usage["cost"] = cost
    return {"choices": [{"message": {"content": text}}], "usage": usage}


def test_cache_key_includes_max_tokens():
    assert cache_key("m", "s", "p", 0.0, 100) != cache_key("m", "s", "p", 0.0, 200)
    assert cache_key("m", "s", "p", 0.0, 100) == cache_key("m", "s", "p", 0.0, 100)


def test_legacy_sqlite_lookup_via_alias(tmp_path, make_client):
    db = tmp_path / "cache" / "llm_cache.sqlite"
    db.parent.mkdir()
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE api_cache (cache_key TEXT PRIMARY KEY, model TEXT, prompt TEXT, system TEXT,"
                 " temperature REAL, response TEXT, latency_ms REAL, created_at REAL)")
    conn.execute("INSERT INTO api_cache (cache_key, response) VALUES (?, ?)",
                 (legacy_cache_key("openai/gpt-4o-mini", "SYS", "PROMPT", 0.0), "pilot answer"))
    conn.commit()
    conn.close()
    before = hashlib.sha256(db.read_bytes()).hexdigest()

    c = make_client("dry_run")
    text, meta = c.call("openai/gpt-4o-mini-2024-07-18", "SYS", "PROMPT", max_tokens=300)
    assert text == "pilot answer"
    assert meta["cached"] and meta["legacy_cache"] and meta["legacy_model"] == "openai/gpt-4o-mini"
    assert c.summary()["calls_uncached"] == 0
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before  # sqlite untouched


def test_dry_run_never_calls_transport(make_client):
    c = make_client("dry_run")  # transport raises if called
    text, meta = c.call(PRICED, "s", "hello world", max_tokens=100)
    assert text == "" and meta["dry_run"]
    c.call(PRICED, "s", "hello world", max_tokens=100)  # same request: would be a cache hit
    s = c.summary()
    assert s["calls_uncached"] == 1 and s["calls_cached"] == 1
    assert s["by_model"][PRICED]["out_tokens_est"] == 50
    assert s["usd_est_total"] > 0


def test_dry_run_unpriced_model(make_client):
    c = make_client("dry_run")
    _, meta = c.call("test/unpriced", "s", "p", max_tokens=10)
    s = c.summary()
    assert meta["cost_usd_est"] is None
    assert s["unpriced_models"] == ["test/unpriced"]
    assert s["by_model"]["test/unpriced"]["usd_est"] is None


def test_execute_requires_key(make_client, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "ROOT_DIR", tmp_path)  # ignore a real .env
    c = make_client("execute", max_cost_usd=1.0)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        c.call(PRICED, "s", "p", max_tokens=10)


def test_execute_requires_max_cost(make_client):
    with pytest.raises(ValueError):
        make_client("execute")


def test_execute_budget_exceeded_and_cost_log(make_client, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-value")
    sent = []

    def fake(payload, key):
        sent.append(payload)
        return _ok(cost=0.6)

    c = make_client("execute", max_cost_usd=1.0, transport=fake)
    c.call(PRICED, "s", "p1", max_tokens=10)
    with pytest.raises(BudgetExceeded):
        c.call(PRICED, "s", "p2", max_tokens=10)
    with pytest.raises(BudgetExceeded):  # already over: refuses before sending
        c.call(PRICED, "s", "p3", max_tokens=10)
    assert len(sent) == 2
    log = [json.loads(x) for x in (tmp_path / "cost.jsonl").read_text().splitlines()]
    assert [r["usd"] for r in log] == [0.6, 0.6] and not log[0]["cached"]
    # second response was cached despite the budget stop -> re-run reuses it
    c2 = make_client("dry_run")
    assert c2.call(PRICED, "s", "p2", max_tokens=10)[0] == "ok"


def test_execute_price_fallback_and_unpriced_refused(make_client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-value")
    c = make_client("execute", max_cost_usd=10.0, transport=lambda p, k: _ok(pt=1_000_000, ct=0))
    _, meta = c.call(PRICED, "s", "p", max_tokens=10)
    assert meta["cost_usd"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="TO_CONFIRM"):
        c.call("test/unpriced", "s", "p", max_tokens=10)


def test_execute_retries_with_backoff(make_client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-value")
    attempts, sleeps = [], []

    def flaky(payload, key):
        attempts.append(1)
        if len(attempts) < 3:
            raise RetryableHTTPError("HTTP 429")
        return _ok()

    c = make_client("execute", max_cost_usd=1.0, transport=flaky, sleep=sleeps.append)
    assert c.call(PRICED, "s", "p", max_tokens=10)[0] == "ok"
    assert len(attempts) == 3 and sleeps == [1, 2]

    always = make_client("execute", max_cost_usd=1.0, sleep=lambda s: None,
                         transport=lambda p, k: (_ for _ in ()).throw(RetryableHTTPError("503")))
    with pytest.raises(RetryableHTTPError):
        always.call(PRICED, "s", "other", max_tokens=10)


def test_transport_maps_429_to_retryable(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, io.BytesIO(b""))
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", boom)
    with pytest.raises(RetryableHTTPError):
        llm_client.openrouter_transport({}, "k")


class ScriptedClient:
    """Fake client returning queued texts and recording prompts."""
    mode = "mock"

    def __init__(self, texts):
        self.texts, self.prompts = list(texts), []

    def call(self, model, system, prompt, max_tokens, temperature=0.0):
        self.prompts.append((prompt, max_tokens))
        return self.texts.pop(0), {"cached": False}


INST = {"question": "Q?", "docs": [{"doc_id": "d0", "text": "alpha"}, {"doc_id": "d1", "text": "beta"}]}


def test_compressor_retries_exactly_once():
    long = "word " * 50
    c = ScriptedClient([long, "short"])
    out = PromptCompressor("C0", ROOT / "prompts" / "c0_naive.txt", "m", c).compress(INST, 10)
    assert out["retried"] and not out["over_budget"] and out["text"] == "short"
    assert len(c.prompts) == 2
    # amendment 2.2-b: tokens are counted on the normalized (stripped) output, so the trailing space is not a token
    assert c.prompts[1][0].endswith(retry_note(count_tokens(long.strip()), 10))
    assert c.prompts[1][0].endswith("dropping distractor content first.")
    assert c.prompts[0][1] == 300  # max(budget*2, 300)

    c = ScriptedClient([long, long, "never used"])
    out = PromptCompressor("C0", ROOT / "prompts" / "c0_naive.txt", "m", c).compress(INST, 10)
    assert out["over_budget"] and len(c.prompts) == 2 and out["meta"]["api_calls"] == 2


def test_examples_block_and_hashes(tmp_path):
    instr = tmp_path / "i.txt"
    instr.write_text("Compress.")
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps([{"question": "EQ", "docs": [{"doc_id": "d0", "text": "et"}],
                               "budget": 7, "output": "EO"}]))
    comp = PromptCompressor("C2", instr, "m", ScriptedClient(["x"]), examples_path=ex)
    prompt = comp.build_prompt(INST, 20)
    assert prompt.startswith("[Examples]\n\n## Example 1\n[Question]\nEQ")
    assert "[Compressed]\nEO" in prompt
    assert prompt.endswith("[Budget]\nYour output must be at most 20 tokens.")
    out = comp.compress(INST, 20)
    assert out["meta"]["instruction_sha256"] == hashlib.sha256(b"Compress.").hexdigest()
    assert out["meta"]["examples_sha256"] == hashlib.sha256(ex.read_bytes()).hexdigest()

    plain = PromptCompressor("C0", instr, "m", ScriptedClient(["x"]))
    assert plain.build_prompt(INST, 20).startswith("[Question]")
    assert plain.compress(INST, 20)["meta"]["examples_sha256"] is None
    with pytest.raises(FileNotFoundError):
        PromptCompressor("C2", instr, "m", None, examples_path=tmp_path / "missing.json")


def test_mock_is_deterministic_and_schema_shaped():
    csr_sys = (ROOT / "prompts" / "judge_csr.txt").read_text()
    a = respond("m", csr_sys, "Q1", 400)
    assert a == respond("m", csr_sys, "Q1", 400)
    keys = {"a_supported", "a_distorted", "b_supported", "b_distorted", "conflict_signaled", "rationale"}
    outcomes = set()
    for i in range(30):
        d = json.loads(respond("m", csr_sys, f"Q{i}", 400))
        assert set(d) == keys
        outcomes.add(d["a_supported"] and d["b_supported"])
    assert outcomes == {True, False}  # varies across inputs

    assert set(json.loads(respond("m", "Return JSON only with false_conflict.", "x", 200))) == {
        "false_conflict", "rationale"}
    d = json.loads(respond("m", "annotator", 'Reply in JSON only: {"category": .., "committed_value": ..}', 120))
    assert d["category"] in {"DEFINITIVE", "CONFLICT_ACK", "INSUFFICIENT", "OTHER"}
    assert d["committed_value"] in {"A", "B", "NONE", "OTHER"}

    prompt = PromptCompressor("C0", ROOT / "prompts" / "c0_naive.txt", "m", None).build_prompt(
        {"question": "Q", "docs": [{"doc_id": "d0", "text": "lorem ipsum " * 200}]}, 30)
    out = respond("m", "Compress the documents.", prompt, 300)
    assert 0 < count_tokens(out) <= 30

    assert respond("m", "Answer the question using only the provided context.", "Context: x", 50)
