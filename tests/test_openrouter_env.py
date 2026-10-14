import os
from pathlib import Path
import pytest
import yaml

from configure_roles import check_family_rule, model_families
from llm_client import LLMClient, load_models_config, MODELS_CONFIG

ROOT_DIR = MODELS_CONFIG.parent.parent


def test_models_yaml_contains_openrouter_phi4_models():
    cfg = load_models_config(MODELS_CONFIG)
    models = cfg["models"]
    prices = cfg["prices"]

    # Check model registration & families
    expected_families = {
        "microsoft/phi-4": "microsoft",
        "meta-llama/llama-3.1-8b-instruct": "meta",
        "openai/gpt-4o-mini": "openai",
        "qwen/qwen-2.5-72b-instruct": "alibaba",
    }
    for m, fam in expected_families.items():
        assert m in models, f"{m} not registered in models.yaml"
        assert models[m].get("family") == fam
        assert models[m].get("provider") == "openrouter"

    # Check exact prices
    assert prices["microsoft/phi-4"] == {"in": 0.07, "out": 0.14}
    assert prices["meta-llama/llama-3.1-8b-instruct"] == {"in": 0.05, "out": 0.08}
    assert prices["openai/gpt-4o-mini"] == {"in": 0.15, "out": 0.60}
    assert prices["qwen/qwen-2.5-72b-instruct"] == {"in": 0.36, "out": 0.40}


def test_models_openrouter_yaml_passes_family_rule():
    openrouter_path = ROOT_DIR / "configs" / "models_openrouter.yaml"
    assert openrouter_path.exists()
    cfg = load_models_config(openrouter_path)
    roles = cfg["roles"]
    fams = model_families(openrouter_path)

    violations = check_family_rule(roles, fams)
    assert violations == [], f"Judge Separation Rule violated: {violations}"
    assert roles["compressor"] == "microsoft/phi-4"
    assert roles["opt_judge"] == "meta-llama/llama-3.1-8b-instruct"
    assert roles["eval_judge"] == "qwen/qwen-2.5-72b-instruct"
    assert roles["downstream"] == "microsoft/phi-4"
    assert roles["reflection"] == "openai/gpt-4o-mini"


def test_ccd_models_config_env_var(monkeypatch, tmp_path):
    custom_cfg = tmp_path / "custom_models.yaml"
    custom_cfg.write_text(yaml.dump({
        "roles": {"compressor": "custom/comp", "downstream": "same_as_compressor"},
        "models": {"custom/comp": {"provider": "openrouter", "family": "custom"}},
    }))

    monkeypatch.setenv("CCD_MODELS_CONFIG", str(custom_cfg))
    cfg = load_models_config()
    assert cfg["roles"]["compressor"] == "custom/comp"
    assert cfg["roles"]["downstream"] == "custom/comp"


def test_llm_client_environment_separation(monkeypatch, tmp_path):
    res_dir = tmp_path / "results_isolated"
    cache_dir = tmp_path / "cache_isolated"
    monkeypatch.setenv("CCD_RESULTS_DIR", str(res_dir))
    monkeypatch.setenv("CCD_CACHE_DIR", str(cache_dir))

    client = LLMClient(mode="dry_run")
    assert client.cache_dir == cache_dir
    assert client.cost_log == res_dir / "api_cost_log.jsonl"
    assert client.manifest_path == res_dir / "models_manifest.json"
