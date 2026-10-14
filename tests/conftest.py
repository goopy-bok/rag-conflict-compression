import pytest

from llm_client import LLMClient

PRICED = "test/priced-model"
CFG = {
    "prices": {PRICED: {"in": 1.0, "out": 2.0}, "test/unpriced": {"in": "TO_CONFIRM", "out": "TO_CONFIRM"}},
    "legacy_aliases": {"openai/gpt-4o-mini": "openai/gpt-4o-mini-2024-07-18"},
}


def _no_network(payload, api_key):
    raise AssertionError("network transport must not be called in this test")


@pytest.fixture
def make_client(tmp_path):
    """LLMClient isolated in tmp_path (own cache dir and cost log, no network by default)."""
    def make(mode="dry_run", **kw):
        kw.setdefault("transport", _no_network)
        kw.setdefault("models_config", CFG)
        kw.setdefault("sleep", lambda s: None)
        return LLMClient(mode=mode, cache_dir=tmp_path / "cache", cost_log=tmp_path / "cost.jsonl", **kw)
    return make
