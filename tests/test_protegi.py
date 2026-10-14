"""ProTeGi port (src/apo_protegi.py). No network: reflection replies come from LLMClient mock mode,
scoring from a fake score_fn (apo_common contract) unless apo_common is importable."""
import json
import os
import random
import re
from pathlib import Path

import pytest

import apo_protegi as P
from budget import count_tokens

# Execution guide v2 (not distributed); the section check is skipped when the file is absent.
GUIDE = Path(os.environ.get("CCD_GUIDE_MD", "docs/execution_guide_v2.md"))


def _inst(i, ctype, a, b=None):
    docs = [{"doc_id": f"d{k}", "text": f"Document {k} about item {i}. " * 20, "role": "distractor"}
            for k in range(3)]
    docs[2].update(text=f"Item {i} was founded in {a}. " * 10, role="conflict_a" if b else "default")
    gold = {"answer_a": a, "span_a": a}
    if b:
        docs[0].update(text=f"Item {i} was founded in {b}. " * 10, role="conflict_b")
        gold.update(answer_b=b, span_b=b)
    return {"id": f"t{i}", "source": "conflictbank", "conflict_type": ctype, "question": f"When was item {i} founded?",
            "docs": docs, "gold": gold, "conflict_positions": [2, 0] if b else [2],
            "entity": f"item {i}", "total_tokens": sum(count_tokens(d["text"]) for d in docs)}


def _split(n_train=6, n_val=4):
    rows = [_inst(i, "none" if i % 3 == 0 else "misinformation", str(1900 + i), None if i % 3 == 0 else str(2000 + i))
            for i in range(n_train + n_val)]
    pairs = [(r, 0.3 if k % 2 else 0.5) for k, r in enumerate(rows)]
    return {"train": pairs[:n_train], "val": pairs[n_train:]}


def fake_score(instruction, items, client, *, examples, task_model, judge_model):
    """Start instruction fails every conflict item; any 'Mock' (edited/paraphrased) instruction passes."""
    good = instruction.startswith("Mock")
    return [{"id": inst["id"], "r": r, "score": 1.0 if good or inst["conflict_type"] == "none" else 0.0,
             "feedback": f"Answer B ('{inst['gold'].get('answer_b')}', from d0) was dropped.",
             "compressed": "short text", "over_budget": False, "out_tokens": 2, "budget": 10}
            for inst, r in items]


# ------------------------------------------------------------------ templates

def _guide_blocks():
    if not GUIDE.exists():
        pytest.skip("execution guide not available")
    sec = GUIDE.read_text(encoding="utf-8").split("### 7.4 ProTeGi")[1].split("### 7.5")[0]
    return {k: re.search(r"\*\*`protegi_" + k + r"\.txt`\*\*.*?\n```\n(.*?)\n```", sec, re.S).group(1)
            for k in ("grad", "edit", "para")}


def test_templates_match_guide_7_4():
    for kind, text in _guide_blocks().items():
        assert P.TEMPLATE_PATHS[kind].read_text(encoding="utf-8") == text + "\n", kind


def test_templates_format_with_expected_fields():
    t = {k: P.load_template(k) for k in P.TEMPLATE_PATHS}
    g = t["grad"].format(prompt="P{x}", error_string="E", num_feedbacks=4)
    assert '"P{x}"' in g and "give 4 reasons" in g and g.endswith("Wrap each reason with <START> and <END>")
    e = t["edit"].format(prompt="P", error_str="E", gradient="G", steps_per_gradient=1)
    assert "problem with this prompt is that G" in e and e.endswith("The 1 new prompts are:")
    assert t["para"].format(prompt_instruction="I").endswith("Input: I\n\nOutput:")


# ------------------------------------------------------------------ parsing / error strings

def test_parse_tagged():
    assert P.parse_tagged("x <START> a <END> y<START>b\nc<END><START>  <END><START>open") == ["a", "b\nc"]
    assert P.parse_tagged("no tags here") == []
    assert P.parse_tagged("<END><START>z<END>") == ["z"]


def test_error_string_format():
    conflict, distractor = _inst(1, "misinformation", "1990", "1995"), _inst(3, "none", "1970")
    res = {"compressed": "Item 1 was founded in 1990 [d2].", "feedback": "Answer B ('1995', from d0) was dropped."}
    s = P.error_string([(conflict, res), (distractor, {**res, "feedback": "Stated a conflict that is not there."})])
    assert s.split("\n")[:5] == [
        "## Example 1",
        "Question: When was item 1 founded?",
        'Expected: a reader of the context can recover both "1990" (from d2) and "1995" (from d0)',
        "Compressed output: Item 1 was founded in 1990 [d2].",
        "Problem: Answer B ('1995', from d0) was dropped.",
    ]
    assert "## Example 2" in s and "Expected: no conflict should be stated" in s
    assert "Document 1 about item" not in s                 # source documents are not included
    assert not re.search(r"\bD[1-7]\b|CCD", s)


# ------------------------------------------------------------------ UCB

def test_ucb_picks_best_arms():
    means = [0.1, 0.9, 0.5, 0.2, 0.85, 0.3, 0.0, 0.4, 0.6, 0.05]
    pulls = [0] * len(means)

    def pull(chosen):
        for i in chosen:
            pulls[i] += 1
        return [means[i] for i in chosen], 5

    q, counts, hist = P.ucb_bandit(len(means), pull, rounds=40, k=3, c=2.0, rng=random.Random(0))
    top = sorted(range(len(means)), key=lambda i: -q[i])[:2]
    assert set(top) == {1, 4}
    assert all(q[i] == pytest.approx(means[i]) for i in range(len(means)) if counts[i])
    assert hist[0]["ucb"] is None and len(hist) == 40
    assert pulls[1] == max(pulls)                         # exploitation concentrates on the best arm


def test_ucb_rounds_budget():
    cfg = dict(P.DEFAULTS)
    assert P.ucb_rounds(36, cfg, 247) == 23                # ceil(25*36 / (8*5))
    assert P.ucb_rounds(9, cfg, 247) == 6                  # ceil(25*9 / 40)
    assert P.ucb_rounds(2, cfg, 3) == 9                    # ceil(25*2 / (2*3))


# ------------------------------------------------------------------ estimate

def test_estimate_calls_defaults():
    e = P.estimate_calls({}, 247, 150)
    # reflection per parent = 4 grad + 16 edit + 32 para; parents = 1 + 5*4
    assert e["by_role"]["reflection"]["calls"] == 52 * 21
    assert e["ucb_rounds_per_step"] == [6, 23, 23, 23, 23, 23]
    items = e["score_items"]
    assert items["minibatch"] == 64 * 21
    assert items["ucb"] == 6 * 8 * 5 + 5 * 23 * 8 * 5
    assert items["val"] == 150 + 6 * 4 * 150
    assert e["by_role"]["compressor"]["calls"] == items["total"] == 1344 + 4840 + 3750
    assert e["by_role"]["compressor"]["calls_max"] == 2 * items["total"]
    assert e["by_role"]["opt_judge"]["calls"] == items["total"]
    no_val = P.estimate_calls({"step_val": False}, 247, 150)["score_items"]["val"]
    assert no_val == 4 * 150


# ------------------------------------------------------------------ end to end (mock)

SMALL = dict(steps=1, minibatch=4, beam=2, max_expansion=2, error_groups=2, errors_per_group=2,
             ucb_samples=2, ucb_arms_per_round=2, budget_per_candidate=4)


def _run(tmp_path, client, split, score_fn, start="Compress the documents."):
    return P.run(SMALL, client, split, start, seed=0, score_fn=score_fn, examples=[],
                 task_model="m/task", judge_model="m/judge", reflection_model="m/reflect",
                 prompts_dir=tmp_path / "prompts", reports_dir=tmp_path / "reports")


def test_mock_end_to_end(tmp_path, make_client):
    client = make_client("mock")
    split = _split()
    log = _run(tmp_path, client, split, fake_score)

    assert log["status"] == "done"
    final = (tmp_path / "prompts" / "c3p.txt").read_text().strip()
    assert final == log["final"]["instruction"] and final.startswith("Mock")   # beam moved off the start
    assert log["final"]["val"] == 1.0 and log["steps"][0]["best_val"] < 1.0 and log["peak_step"] == 1

    saved = json.loads((tmp_path / "reports" / "apo_log_C3-P_seed0.json").read_text())
    assert saved["final"]["instruction"] == final and len(saved["steps"]) == 2
    assert len(saved["steps"][1]["beam"]) == 2

    trace = [json.loads(x) for x in (tmp_path / "reports" / "protegi_trace.jsonl").read_text().splitlines()]
    assert [t["type"] for t in trace] == ["val", "expand", "select", "val"]
    exp = trace[1]
    assert exp["n_errors"] > 0 and len(exp["groups"]) == 2
    assert all(len(g["gradients"]) == 4 and g["gradient_raw"] for g in exp["groups"])
    assert len(exp["candidates_kept"]) == 2 and trace[2]["candidates"][0]["from_beam"]

    # observed reflection calls match the estimate formula for this config (mock gives 4 reasons, 1 edit)
    est = P.estimate_calls(SMALL, len(split["train"]), len(split["val"]))
    assert sum(log["reflection_calls"].values()) == est["by_role"]["reflection"]["calls"]
    assert log["client_summary"]["by_model"]["m/reflect"]["calls"] == est["by_role"]["reflection"]["calls"]
    assert log["metric_items_evaluated"] <= est["score_items"]["total"]
    assert log["parse_failures"] == {"grad": 0, "edit": 0, "para": 0}


def test_no_errors_means_no_expansion(tmp_path, make_client):
    perfect = lambda ins, items, c, **kw: [{**r, "score": 1.0} for r in fake_score(ins, items, c, **kw)]  # noqa: E731
    log = _run(tmp_path, make_client("mock"), _split(), perfect)
    assert log["reflection_calls"] == {"grad": 0, "edit": 0, "para": 0}
    assert log["final"]["instruction"] == "Compress the documents."


def test_unparseable_reflection_counts_failure(tmp_path, make_client, monkeypatch):
    monkeypatch.setattr(P, "_mock_reflection", lambda kind, prompt: "no tags at all")
    log = _run(tmp_path, make_client("mock"), _split(), fake_score)
    assert log["parse_failures"]["grad"] == 2 and log["reflection_calls"]["edit"] == 0
    assert log["final"]["instruction"] == "Compress the documents."


def test_mock_end_to_end_with_apo_common(tmp_path, make_client):
    apo_common = pytest.importorskip("apo_common")
    log = _run(tmp_path, make_client("mock"), _split(), apo_common.score_instruction)
    assert log["status"] == "done" and (tmp_path / "prompts" / "c3p.txt").exists()
    trace = [json.loads(x) for x in (tmp_path / "reports" / "protegi_trace.jsonl").read_text().splitlines()]
    for g in trace[1]["groups"]:
        assert not re.search(r"\bD[1-7]\b|CCD", g["error_string"])


def test_cli_mock_and_execute_guard(tmp_path, monkeypatch, capsys):
    import sys
    import types
    fake = types.ModuleType("apo_common")
    fake.load_apo_split = lambda seed=0: _split()
    fake.score_instruction = fake_score
    fake.c2_freeze_warning = lambda path: "WARNING: C2 is not frozen"
    monkeypatch.setitem(sys.modules, "apo_common", fake)
    monkeypatch.setattr(P, "PROMPTS_DIR", tmp_path / "prompts")
    monkeypatch.setattr(P, "REPORTS_DIR", tmp_path / "reports")

    with pytest.raises(SystemExit, match="max-cost"):
        P.main(["--mode", "execute"])
    P.main(["--mode", "mock", "--steps", "1", "--minibatch", "4", "--beam", "2", "--max-expansion", "2",
            "--error-groups", "1", "--ucb-samples", "2", "--ucb-arms-per-round", "2"])
    assert (tmp_path / "prompts" / "mock" / "c3p.txt").exists()
    log = json.loads((tmp_path / "reports" / "mock" / "apo_log_C3-P_seed0.json").read_text())
    assert log["status"] == "done" and log["mode"] == "mock" and log["config"]["beam"] == 2
    assert "C2 is not frozen" in capsys.readouterr().err and log["c2_freeze_warning"]
