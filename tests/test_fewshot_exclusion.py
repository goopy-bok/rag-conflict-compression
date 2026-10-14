import json

import pytest

import run

# Needs data/processed/dev.jsonl, which is not redistributed (rebuild with src/data_build.py).
pytestmark = pytest.mark.skipif(not run.SPLITS["dev"].exists(), reason="dev data not built (see README: Rebuilding the data)")


def test_fewshot_ids_are_dropped_from_dev():
    held_out = run.fewshot_source_ids()
    assert held_out == {"dev_0093", "dev_0070", "dev_0024"}
    dev_ids = {r["id"] for r in run.load_split("dev")}
    assert not dev_ids & held_out
    assert len(dev_ids) == 297


def test_fewshot_outputs_fit_budget_and_keep_both_answers():
    from budget import count_tokens
    examples = json.loads(run.FEWSHOT_FILE.read_text(encoding="utf-8"))
    dev = {r["id"]: r for r in map(json.loads, (run.SPLITS["dev"]).read_text(encoding="utf-8").splitlines())}
    for ex in examples:
        assert count_tokens(ex["output"]) <= ex["budget"]
        gold = dev[ex["source_id"]]["gold"]
        assert gold["answer_a"] in ex["output"] and gold["answer_b"] in ex["output"]
