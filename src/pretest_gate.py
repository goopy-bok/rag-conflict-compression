"""
Pre-test gate (amendment 2.2-c). Runs after GEPA and before freeze/test: the sealed test splits are evaluated
once, so the pipeline stops here if a test prompt or the optimized prompts are unusable. Never reads a test split.

  uv run python src/pretest_gate.py                          # results/pretest_gate.json; exit 0 pass, 1 fail
  uv run python src/pretest_gate.py --allow-no-improvement   # K3 failure recorded as an accepted override

Checks (all must pass):
  K1 G0-auto passed (amendment 2.2-d): results/g0_local/g0_auto.json has "pass": true (inter-judge kappa and
     negative control; accuracy vs the author labels is checked at the end).
  K2 GEPA finished: reports/apo_log_<C3-G1|C3-G>_seed0.json has status "ok" and its prompt file exists.
  K3 the optimizer moved C1 (preregistration checkpoint 3): C3-G1 best val score > the val score of its start
     candidate (C1) on the same val set. GEPA's best includes the start candidate, so ">=" would always hold.
     C3-G vs its start (C2) is recorded only (a lower C3-G is an expected case in plan v2).
  K4 few-shot fixed: fewshot_check.check_log is valid for both runs.
  K5 format: each test prompt (C0 C1 C2 C3-G1 C3-G) compresses the first --n dev items (default 100, r 0.3) with
     the compressor role; per prompt M1 >= 0.95 and M2' >= 0.90 (amendment 2.2-e; normalization N,
     pilot_select.format_flags_v2 with C2's tagged ANSWER line accepted), and content (amendment 2.2-g): the share
     of items whose output keeps a gold answer (apo_common.answer_kept, any of answer_a/answer_b) is at least
     KEPT_REL x that of C0. The format flags alone pass degenerate text once N has cut it short.
     Compression only (no judge, no downstream); calls go through the LLM cache, so a rerun is cheap.
     --accept-k5 C2 (or GATE_ACCEPT_K5=C2; amendment 2.2-g decision, 2026-10-10): C2 v0's K5 failure is recorded
     as a result, not blocking. C3-G (optimized from C2) inherits it only if none of its M1, M2', kept is below
     C2's; a C3-G worse than its start still stops the gate.
  K6 seal intact and test unopened: seal.verify_sealed() passes and results/test_access_log.jsonl has no
     "allowed": true entry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from apo_common import answer_kept  # noqa: E402
from budget import compute_budget  # noqa: E402
from fewshot_check import check_log  # noqa: E402
from pilot_select import format_flags_v2  # noqa: E402
from run import CONDITIONS_CONFIG, build_compressor, load_split  # noqa: E402
from seal import SealError, verify_sealed  # noqa: E402

TEST_PROMPTS = ["C0", "C1", "C2", "C3-G1", "C3-G"]
GEPA_RUNS = ["C3-G1", "C3-G"]
R, SEED = 0.3, 0
# Amendment 2.2-e (before the gate ran): n 30 -> 100 and M2' 0.95 -> 0.90. With the selected compressor's base
# anomaly rate (phi3.5: 10/200 QA mode in the head-to-head), n=30 and M2' >= 0.95 fails by chance with
# P = 0.948 over 5 prompts; n=100 and M2' >= 0.90 gives P = 0.056 and still stops a prompt at >= 11% anomalies.
N_FORMAT = 100
M1_MIN, M2_MIN = 0.95, 0.90
# Amendment 2.2-g (set after the first gate's dev outputs: kept C0 0.88, C1 0.91, C2 v0 0.42, degenerate C3-G 0.10;
# the test split is unseen). A prompt that keeps fewer than 80% as many answers as the naive prompt is broken.
KEPT_REL = 0.8
K5_INHERITS = {"C3-G": "C2"}   # GEPA start of each optimized prompt that uses an accept-able start
EPS = 1e-9
OUT = ROOT_DIR / "results" / "pretest_gate.json"
FORMAT_OUT = ROOT_DIR / "results" / "pretest_gate_format.jsonl"
GATE_CACHE = ROOT_DIR / "results" / "pretest_gate_cache"


def check_g0(path: Path = ROOT_DIR / "results" / "g0_local" / "g0_auto.json") -> dict:
    if not path.exists():
        return {"ok": False, "reason": f"{path} missing"}
    return {"ok": bool(json.loads(path.read_text(encoding="utf-8")).get("pass")), "path": str(path)}


def gepa_log_path(condition: str, root: Path = ROOT_DIR) -> Path:
    return root / "reports" / f"apo_log_{condition}_seed{SEED}.json"


def check_gepa_finished(logs: dict[str, dict | None], root: Path = ROOT_DIR) -> dict:
    bad = {}
    for cond, log in logs.items():
        if log is None:
            bad[cond] = "log missing"
        elif log.get("status") != "ok":
            bad[cond] = f"status {log.get('status')}"
        else:
            p = Path(log.get("final_instruction_path") or "")
            if not (p if p.is_absolute() else root / p).exists():
                bad[cond] = f"prompt file missing: {p}"
    return {"ok": not bad, "problems": bad}


def start_vs_best(log: dict) -> dict:
    """Val score of the start candidate (index 0 = the start instruction) vs the selected best candidate."""
    start = float(log["candidates"][0]["val_score"])
    best = float(log["best_val_score"])
    return {"start_val": start, "best_val": best, "best_idx": log.get("best_idx"), "improved": best > start + EPS}


def check_optimizer_moved(logs: dict[str, dict | None], allow_no_improvement: bool = False) -> dict:
    c3g1 = logs.get("C3-G1")
    if not c3g1 or c3g1.get("status") != "ok":
        return {"ok": False, "reason": "C3-G1 log missing or not ok"}
    res = {"C3-G1": start_vs_best(c3g1)}
    if logs.get("C3-G") and logs["C3-G"].get("status") == "ok":
        res["C3-G (recorded only)"] = start_vs_best(logs["C3-G"])
    improved = res["C3-G1"]["improved"]
    res["ok"] = improved or allow_no_improvement
    if not improved:
        res["note"] = ("C3-G1 did not beat its C1 start on val; "
                       + ("accepted by override (--allow-no-improvement)" if allow_no_improvement
                          else "stop here and decide (rerun with --allow-no-improvement to test anyway)"))
    return res


def check_fewshot(logs: dict[str, dict | None]) -> dict:
    runs = {c: check_log(log) for c, log in logs.items() if log}
    return {"ok": len(runs) == len(logs) and all(r["valid"] for r in runs.values()),
            "runs": {c: {"valid": r["valid"], "examples_fixed": r["examples_input_fixed"]["ok"],
                         "demos_empty": r["demos_empty"]["ok"]} for c, r in runs.items()}}


def check_format(client, model: str, instances: list[dict], conditions=TEST_PROMPTS,
                 conditions_config: Path = CONDITIONS_CONFIG, out: Path | None = FORMAT_OUT) -> dict:
    per, rows = {}, []
    for cond in conditions:
        comp = build_compressor(cond, SEED, client, model, conditions_config=conditions_config)
        recs = []
        for inst in instances:
            budget = compute_budget(inst["total_tokens"], R)
            c = comp.compress(inst, budget)
            fl = format_flags_v2(c["text"], tagged_answer_ok=True)  # C2's 'ANSWER: [dX] <value>' line is valid
            kept = any(answer_kept(inst["gold"].get(k), c["text"]) for k in ("answer_a", "answer_b"))
            recs.append({"condition": cond, "id": inst["id"], "budget": budget, "out_tokens": c["tokens"],
                         "over_budget": c["over_budget"], "ok": fl["ok"], "kept": kept, "flags": fl, "text": c["text"]})
        n = len(recs)
        m1 = sum(not r["over_budget"] for r in recs) / n
        m2 = sum(r["ok"] for r in recs) / n
        per[cond] = {"n": n, "M1": m1, "M2p": m2, "kept": sum(r["kept"] for r in recs) / n,
                     "ok": m1 >= M1_MIN - EPS and m2 >= M2_MIN - EPS,
                     "qa_mode": sum(bool(r["flags"].get("qa_mode")) for r in recs),
                     "json_left": sum(bool(r["flags"].get("json_left")) for r in recs),
                     "empty": sum(r["flags"]["empty"] for r in recs)}
        rows += recs
    if "C0" in per:
        kept_min = KEPT_REL * per["C0"]["kept"]
        for v in per.values():
            v["kept_min"] = kept_min
            v["ok"] = v["ok"] and v["kept"] >= kept_min - EPS
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"ok": all(v["ok"] for v in per.values()), "per_prompt": per, "model": model, "n_items": len(instances)}


def accept_k5(k5: dict, accept: tuple[str, ...]) -> dict:
    """Mark accepted K5 failures (raw "ok" is kept) and recompute the K5 verdict."""
    per = k5.get("per_prompt")
    if not accept or not per:
        return k5
    for cond, v in per.items():
        src = K5_INHERITS.get(cond)
        if v["ok"]:
            continue
        if cond in accept:
            v["accepted"] = "override"
        elif src in accept and src in per and all(v[m] >= per[src][m] - EPS for m in ("M1", "M2p", "kept")):
            v["accepted"] = f"inherits {src} override: no K5 value below {src}"
    k5["accept"] = list(accept)
    k5["ok"] = all(v["ok"] or v.get("accepted") for v in per.values())
    return k5


def check_test_unopened(root: Path = ROOT_DIR) -> dict:
    try:
        verify_sealed(root)
    except SealError as e:
        return {"ok": False, "reason": f"seal: {e}"}
    log = root / "results" / "test_access_log.jsonl"
    opened = []
    if log.exists():
        opened = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        opened = [e for e in opened if e.get("allowed")]
    return {"ok": not opened, "allowed_accesses": len(opened)}


def run_gate(client, model: str, *, n: int = N_FORMAT, allow_no_improvement: bool = False,
             accept: tuple[str, ...] = (), root: Path = ROOT_DIR) -> dict:
    logs = {c: (json.loads(p.read_text(encoding="utf-8")) if (p := gepa_log_path(c, root)).exists() else None)
            for c in GEPA_RUNS}
    checks = {"K1_g0_auto": check_g0(root / "results" / "g0_local" / "g0_auto.json"),
              "K2_gepa_finished": check_gepa_finished(logs, root),
              "K3_optimizer_moved_C1": check_optimizer_moved(logs, allow_no_improvement),
              "K4_fewshot_fixed": check_fewshot(logs),
              "K6_test_unopened": check_test_unopened(root)}
    # K5 needs the C3 prompt files; skip the compressions when the GEPA runs are not usable
    checks["K5_format"] = (accept_k5(check_format(client, model, load_split("dev", limit=n)), accept)
                           if checks["K2_gepa_finished"]["ok"] else {"ok": False, "reason": "skipped: K2 failed"})
    return {"pass": all(c["ok"] for c in checks.values()), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "allow_no_improvement": allow_no_improvement, "accept_k5": list(accept), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_FORMAT, help=f"dev items per prompt for K5 (default {N_FORMAT})")
    ap.add_argument("--allow-no-improvement", action="store_true")
    ap.add_argument("--accept-k5", default=os.environ.get("GATE_ACCEPT_K5", ""),
                    help="comma list of test prompts whose K5 failure is accepted (env GATE_ACCEPT_K5)")
    a = ap.parse_args(argv)
    accept = tuple(c for c in a.accept_k5.split(",") if c.strip())
    if bad := [c for c in accept if c not in TEST_PROMPTS]:
        ap.error(f"--accept-k5: unknown prompt(s) {bad}; choose from {TEST_PROMPTS}")
    from llm_client import LLMClient, load_models_config
    model = load_models_config()["roles"]["compressor"]
    # Own cache (amendment 2.2-e addendum): cache keys do not include the Ollama version, so the shared cache would
    # serve the pre-upgrade (0.30.8) pilot outputs for dev C0/C1; K5 must judge the current runtime only.
    GATE_CACHE.mkdir(parents=True, exist_ok=True)
    client = LLMClient(mode="execute", max_cost_usd=0.0, cache_dir=GATE_CACHE)
    res = run_gate(client, model, n=a.n, allow_no_improvement=a.allow_no_improvement, accept=accept)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res["checks"].items():
        print(f"{k:24s} {'PASS' if v['ok'] else 'FAIL'}  {v.get('reason') or v.get('note') or ''}")
    print(f"pretest gate: {'PASS' if res['pass'] else 'FAIL'} -> {OUT}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
