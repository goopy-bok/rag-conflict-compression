"""Amendment 2.2-c: pre-test gate (src/pretest_gate.py), preflight (src/preflight.py), run_lite.sh wiring.
Fake transports and temporary roots only; no network, no test split."""
import json
from pathlib import Path

import pytest

import preflight as pf
import pretest_gate as pg
from configure_roles import assign_roles
from fewshot_check import expected_examples_block, sha256_text
from seal import seal_files
from test_e2e_mock import _inst
from test_local import LOCAL_CFG, FakeOllama

ROOT = Path(__file__).resolve().parent.parent


# ---------- K2 / K3 ----------
def _log(cond="C3-G1", start=0.40, best=0.55, best_idx=3, status="ok", prompt=None, uses_fewshot=False):
    return {"condition": cond, "seed": 0, "status": status, "best_val_score": best, "best_idx": best_idx,
            "candidates": [{"idx": 0, "val_score": start}, {"idx": best_idx, "val_score": best}],
            "final_instruction_path": str(prompt) if prompt else None, "uses_fewshot": uses_fewshot,
            "examples_sha256_seen": [sha256_text(expected_examples_block(uses_fewshot))], "program_path": None}


def test_optimizer_moved_requires_strict_improvement():
    assert pg.check_optimizer_moved({"C3-G1": _log()})["ok"] is True
    res = pg.check_optimizer_moved({"C3-G1": _log(start=0.5, best=0.5, best_idx=0)})   # GEPA kept the start (C1)
    assert res["ok"] is False and res["C3-G1"]["improved"] is False and "decide" in res["note"]
    res = pg.check_optimizer_moved({"C3-G1": _log(start=0.5, best=0.5, best_idx=0)}, allow_no_improvement=True)
    assert res["ok"] is True and "override" in res["note"]


def test_c3g_below_c2_is_recorded_not_blocking():
    res = pg.check_optimizer_moved({"C3-G1": _log(), "C3-G": _log("C3-G", start=0.6, best=0.6, best_idx=0)})
    assert res["ok"] is True and res["C3-G (recorded only)"]["improved"] is False


def test_gepa_finished(tmp_path):
    p = tmp_path / "c3g1_seed0.txt"
    p.write_text("x")
    assert pg.check_gepa_finished({"C3-G1": _log(prompt=p)}, tmp_path)["ok"] is True
    bad = pg.check_gepa_finished({"C3-G1": None, "C3-G": _log("C3-G", status="error")}, tmp_path)
    assert bad["ok"] is False and set(bad["problems"]) == {"C3-G1", "C3-G"}
    assert pg.check_gepa_finished({"C3-G1": _log(prompt=tmp_path / "missing.txt")}, tmp_path)["ok"] is False


# ---------- K4 ----------
def test_fewshot_check_uses_existing_validator():
    logs = {"C3-G1": _log(), "C3-G": _log("C3-G", uses_fewshot=True)}
    assert pg.check_fewshot(logs)["ok"] is True
    logs["C3-G"]["examples_sha256_seen"] = ["something-else"]                       # examples were altered
    assert pg.check_fewshot(logs)["ok"] is False
    assert pg.check_fewshot({"C3-G1": _log(), "C3-G": None})["ok"] is False


# ---------- K5 ----------
def _conditions(tmp_path) -> Path:
    paths = {}
    for c in pg.TEST_PROMPTS:
        p = tmp_path / f"{c}.txt"
        p.write_text("Compress the documents. Output only the compressed context.")
        paths[c] = {"instruction": str(p)}
    cfg = tmp_path / "conditions.yaml"
    cfg.write_text(json.dumps(paths))                                                # JSON is valid YAML
    return cfg


@pytest.mark.parametrize("text,ok", [("[d0] Item founded in 1990. [d1] 1995.", True),
                                     ("CONFLICT: [d0] 1990 vs [d1] 1995\n[d0] says 1990.", True),   # C2 format
                                     ("ANSWER: [d0] 1990\n[d0] says 1990.", True),  # C2 no-conflict line
                                     ("Answer: 1990\nContext: founded then.", False),  # untagged answer = QA
                                     ("{}", False),                                 # empty after normalization N
                                     ("Question: when?\nAnswer: 1990", False)])     # QA mode
def test_format_check(make_client, tmp_path, text, ok):
    c = make_client("execute", models_config=LOCAL_CFG, ollama_transport_fn=FakeOllama(text=text), max_cost_usd=0.0)
    insts = [_inst(1, "misinformation", "1990", "1995"), _inst(2, "temporal", "2001", "2011")]
    res = pg.check_format(c, "qwen2.5:7b", insts, conditions_config=_conditions(tmp_path), out=tmp_path / "f.jsonl")
    assert res["ok"] is ok and set(res["per_prompt"]) == set(pg.TEST_PROMPTS)
    assert len((tmp_path / "f.jsonl").read_text().splitlines()) == 2 * len(pg.TEST_PROMPTS)


class _GarbageFor(FakeOllama):
    """Degenerate text when the system prompt contains `marker` (2.2-g: format flags pass it, content check stops it)."""
    def __init__(self, marker, **kw):
        super().__init__(**kw)
        self.marker = marker

    def __call__(self, path, payload):
        r = super().__call__(path, payload)
        if path != "/api/tags" and self.marker in json.dumps(payload["messages"]):
            r["message"]["content"] = "B\n\nQ Q\n\nSolution"
        return r


def test_format_check_stops_prompt_that_keeps_no_answers(make_client, tmp_path):
    cfg = _conditions(tmp_path)
    Path(json.loads(cfg.read_text())["C3-G"]["instruction"]).write_text("Ramble about quizzes.")
    fake = _GarbageFor("Ramble", text="[d0] Item founded in 1990. [d1] 1995.")
    c = make_client("execute", models_config=LOCAL_CFG, ollama_transport_fn=fake, max_cost_usd=0.0)
    insts = [_inst(1, "misinformation", "1990", "1995"), _inst(2, "temporal", "2001", "2011")]
    res = pg.check_format(c, "qwen2.5:7b", insts, conditions_config=cfg, out=None)
    per = res["per_prompt"]
    assert res["ok"] is False and per["C3-G"]["ok"] is False and per["C3-G"]["M2p"] == 1.0   # format alone passes it
    assert per["C3-G"]["kept"] == 0.0 and all(per[k]["ok"] for k in ("C0", "C1", "C2", "C3-G1"))


def _k5(c2, c3g):
    ok = {"M1": 1.0, "M2p": 0.95, "kept": 0.9, "ok": True}
    return {"ok": False, "per_prompt": {"C0": dict(ok), "C1": dict(ok), "C3-G1": dict(ok),
                                        "C2": {**dict(zip(("M1", "M2p", "kept"), c2)), "ok": False},
                                        "C3-G": {**dict(zip(("M1", "M2p", "kept"), c3g)), "ok": False}}}


def test_accept_k5_c2_override_and_c3g_inheritance():
    assert pg.accept_k5(_k5((0.9, 0.8, 0.3), (0.9, 0.8, 0.3)), ())["ok"] is False             # no override
    res = pg.accept_k5(_k5((0.9, 0.8, 0.3), (0.92, 0.85, 0.3)), ("C2",))                     # C3-G no worse
    assert res["ok"] is True and res["per_prompt"]["C2"]["accepted"] == "override"
    assert res["per_prompt"]["C2"]["ok"] is False and "inherits C2" in res["per_prompt"]["C3-G"]["accepted"]
    res = pg.accept_k5(_k5((0.9, 0.8, 0.3), (0.99, 0.96, 0.1)), ("C2",))                     # degenerate C3-G
    assert res["ok"] is False and "accepted" not in res["per_prompt"]["C3-G"]


def test_accept_k5_cli_rejects_unknown_prompt(monkeypatch):
    monkeypatch.setenv("GATE_ACCEPT_K5", "C9")
    with pytest.raises(SystemExit):
        pg.main([])


# ---------- K6 ----------
def _sealed_root(tmp_path) -> Path:
    (tmp_path / "data" / "sealed").mkdir(parents=True)
    f = tmp_path / "data" / "sealed" / "test_id_pool.jsonl"
    f.write_text('{"id": "x"}\n')
    seal_files([f], root=tmp_path)
    return tmp_path


def test_test_unopened(tmp_path):
    root = _sealed_root(tmp_path)
    assert pg.check_test_unopened(root)["ok"] is True
    log = root / "results" / "test_access_log.jsonl"
    log.parent.mkdir()
    log.write_text(json.dumps({"split": "test_id", "allowed": False}) + "\n")         # denied attempts do not count
    assert pg.check_test_unopened(root)["ok"] is True
    log.write_text(log.read_text() + json.dumps({"split": "test_id", "allowed": True}) + "\n")
    assert pg.check_test_unopened(root) == {"ok": False, "allowed_accesses": 1}
    (root / "data" / "sealed" / "test_id_pool.jsonl").write_text('{"id": "changed"}\n')
    assert pg.check_test_unopened(root)["ok"] is False                              # seal broken


def test_preflight_seal_state_allows_resume_only_after_gate_and_freeze(tmp_path):
    root = _sealed_root(tmp_path)
    log = root / "results" / "test_access_log.jsonl"
    log.parent.mkdir()
    log.write_text(json.dumps({"split": "test_id", "allowed": True}) + "\n")
    assert pf.check_seal_state(root)["ok"] is False                                 # opened without gate/freeze
    (root / "results" / "pretest_gate.json").write_text(json.dumps({"pass": True}))
    assert pf.check_seal_state(root)["ok"] is False                                 # not frozen yet
    (root / "results" / "FROZEN_PROMPTS.json").write_text("{}")
    res = pf.check_seal_state(root)
    assert res["ok"] is True and "resume" in res["note"]
    (root / "data" / "sealed" / "test_id_pool.jsonl").write_text("tampered\n")
    assert pf.check_seal_state(root)["ok"] is False                                 # a broken seal never passes


def test_run_lite_gate_is_skipped_once_passed():
    sh = (ROOT / "scripts" / "run_lite.sh").read_text(encoding="utf-8")
    gate = sh.split("phase_gate() {")[1].split("\n}")[0]
    assert "if gate_passed; then" in gate and "return 0" in gate


# ---------- preflight ----------
def test_parse_swap_and_resources():
    s = pf.parse_swap("total = 26624.00M  used = 26130.19M  free = 493.81M  (encrypted)")
    assert s == {"total_mb": 26624.0, "used_mb": 26130.19}
    full = "total = 1000.00M  used = 950.00M  free = 50.00M"
    assert pf.check_resources(swap_text=full, free_bytes=100 * 10**9)["ok"] is False
    assert pf.check_resources(allow_swap=True, swap_text=full, free_bytes=100 * 10**9)["ok"] is True
    assert pf.check_resources(swap_text="total = 0.00M  used = 0.00M", free_bytes=10 * 10**9)["ok"] is False  # disk


def test_roles_must_match_role_table():
    roles = {**assign_roles("phi3.5"), "downstream": "phi3.5"}
    assert pf.check_roles({"roles": roles})["ok"] is True
    assert pf.check_roles({"roles": {**roles, "eval_judge": "qwen2.5:14b"}})["ok"] is False   # stale table
    assert pf.check_roles({"roles": {**roles, "compressor": "nope"}})["ok"] is False


def test_pilot_input(tmp_path):
    for c in ("C0", "C1"):
        (tmp_path / f"gemma4-e4b_{c}.jsonl").write_text("{}\n" * 25)
    assert pf.check_pilot("gemma4:e4b", tmp_path)["ok"] is True
    (tmp_path / "gemma4-e4b_C1.jsonl").write_text("{}\n" * 24)
    assert pf.check_pilot("gemma4:e4b", tmp_path)["ok"] is False


def _show_transport(thinking_default):
    def send(path, payload):
        if path == "/api/tags":
            return {"models": [{"name": "gemma4:e4b"}, {"name": "phi3.5:latest"}]}
        if path == "/api/version":
            return {"version": "0.40.2"}
        if payload["model"] == "gemma4:e4b":
            return {"capabilities": ["completion", "thinking"], "thinking": {"values": [False, True],
                                                                           "default": thinking_default}}
        return {"capabilities": ["completion"]}
    return send


def test_ollama_thinking_must_be_configured():
    cfg = {"roles": {"compressor": "gemma4:e4b", "downstream": "gemma4:e4b", "opt_judge": "phi3.5"},
           "models": {"gemma4:e4b": {"provider": "ollama"}, "phi3.5": {"provider": "ollama"}}}
    res = pf.check_ollama(cfg, transport=_show_transport(True))
    assert res["ok"] is False and res["thinking_default_on_without_think"] == ["gemma4:e4b"]
    cfg["models"]["gemma4:e4b"]["think"] = False
    assert pf.check_ollama(cfg, transport=_show_transport(True))["ok"] is True
    cfg["roles"]["eval_judge"] = "missing-model"
    cfg["models"]["missing-model"] = {"provider": "ollama"}
    assert pf.check_ollama(cfg, transport=_show_transport(True))["missing"] == ["missing-model"]


# ---------- run_lite.sh wiring ----------
def test_run_lite_order_and_gate_requirements():
    sh = (ROOT / "scripts" / "run_lite.sh").read_text(encoding="utf-8")
    assert "phases=(preflight g0auto gepa gate freeze test stats)" in sh                  # amendment 2.2-d
    freeze = sh.split("phase_freeze() {")[1].split("\n}")[0]
    test = sh.split("phase_test() {")[1].split("\n}")[0]
    assert "require_gate freeze" in freeze and "require_gate test" in test
    assert 'PILOT_DIR=${PILOT_DIR:-results/pilot_h2h}' in sh and '"$PILOT_DIR/${stem}_C0.jsonl"' in sh
