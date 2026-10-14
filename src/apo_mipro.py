"""
C3-M: MIPROv2 over the compressor instruction (execution guide v2 section 7.3).

  uv run python src/apo_mipro.py --mode mock
  uv run python src/apo_mipro.py --mode execute --max-cost 20
  uv run python src/apo_mipro.py --estimate

Same program, split and score as GEPA (dspy_adapter.BudgetedCompressor, apo_common score; float only).
dspy.MIPROv2(metric, prompt_model, task_model, auto="medium", seed=0, max_bootstrapped_demos=0,
max_labeled_demos=0): zero-shot, so MIPROv2 proposes instructions only; the fixed few-shot block
still reaches the compressor through the `examples` input. prompt_model = ClientLM(reflection role,
temperature 1.0; MIPROv2 sets its own init_temperature=1.0 per proposal), task_model =
ClientLM(compressor role). MIPROv2 step 3 needs the optional package optuna.
Start = --start-instruction (default prompts/c2_best.txt). Outputs: prompts/c3m.txt (+ c3m.json),
reports/apo_log_C3-M_seed0.json; mock -> prompts/mock/, reports/mock/.
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

import dspy  # noqa: E402

import apo_common as ac  # noqa: E402
import dspy_adapter as da  # noqa: E402
from llm_client import LLMClient, load_models_config  # noqa: E402

CONDITION = "C3-M"
DEFAULT_START = ROOT_DIR / "prompts" / "c2_best.txt"


def output_paths(mode: str, prompts_dir: Path | None = None, reports_dir: Path | None = None) -> dict[str, Path]:
    prompts_base = Path(prompts_dir or os.environ.get("CCD_PROMPTS_DIR") or ROOT_DIR / "prompts")
    reports_base = Path(reports_dir or os.environ.get("CCD_REPORTS_DIR") or ROOT_DIR / "reports")
    sub = "mock" if mode == "mock" else ""
    return {"prompt": prompts_base / sub / "c3m.txt", "program": prompts_base / sub / "c3m.json",
            "log": reports_base / sub / f"apo_log_{CONDITION}_seed0.json"}


def run_mipro(client, *, start_instruction: Path = DEFAULT_START, split: dict | None = None,
              auto: str | None = "medium", num_candidates: int | None = None, num_trials: int | None = None,
              minibatch: bool = True, out: dict[str, Path] | None = None, seed: int = 0) -> dict:
    try:
        import optuna  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit("MIPROv2 needs the optional package 'optuna' (not installed in this environment)") from e
    roles = load_models_config()["roles"]
    split = split or ac.load_apo_split(0)
    out = out or output_paths(client.mode)
    start_instruction = Path(start_instruction)
    examples_block = ac.expected_examples_block(True)
    instances = {inst["id"]: inst for part in ("train", "val") for inst, _ in split[part]}

    task_lm = da.ClientLM(client, roles["compressor"], temperature=0.0)
    prompt_lm = da.ClientLM(client, roles["reflection"], temperature=1.0, max_tokens=4096)
    mipro = dspy.MIPROv2(metric=da.make_metric(instances, client, roles["opt_judge"], feedback=False),
                         prompt_model=prompt_lm, task_model=task_lm, auto=auto, num_candidates=num_candidates,
                         seed=seed, num_threads=client.concurrency, track_stats=True)
    ac.reset_examples_seen()
    t0 = time.time()
    with dspy.context(lm=task_lm):
        optimized = mipro.compile(da.BudgetedCompressor(start_instruction.read_text(encoding="utf-8"), examples_block),
                                  trainset=da.to_examples(split["train"]), valset=da.to_examples(split["val"]),
                                  max_bootstrapped_demos=0, max_labeled_demos=0, num_trials=num_trials,
                                  minibatch=minibatch)
    errors = task_lm.errors + prompt_lm.errors
    budget_hit = client.max_cost_usd is not None and client.usd_spent >= client.max_cost_usd > 0  # local runs: cap 0, never hit
    status = "error" if errors else ("budget_exceeded" if budget_hit else "ok")

    trials = []
    for n, t in sorted(getattr(optimized, "trial_logs", {}).items()):
        prog = t.get("full_eval_program") or t.get("mb_program")
        trials.append({"trial": n, "full_eval_score": t.get("full_eval_score"), "mb_score": t.get("mb_score"),
                       "instruction": da.program_instruction(prog) if prog else None,
                       "total_eval_calls_so_far": t.get("total_eval_calls_so_far")})
    full = [t for t in trials if t["full_eval_score"] is not None]
    best_full = max(full, key=lambda t: t["full_eval_score"]) if full else None
    log = {
        "condition": CONDITION, "seed": seed, "mode": client.mode, "status": status, "errors": errors[:20],
        "task_model": roles["compressor"], "prompt_model": roles["reflection"], "judge_model": roles["opt_judge"],
        "start_instruction_path": str(start_instruction),
        "start_instruction_sha256": ac.sha256_text(start_instruction.read_text(encoding="utf-8")),
        "c2_freeze_warning": ac.c2_freeze_warning(start_instruction), "uses_fewshot": True,
        "examples_sha256_expected": ac.sha256_text(examples_block), "examples_sha256_seen": ac.examples_seen(),
        "final_instruction_path": str(out["prompt"]) if status == "ok" else None,
        "program_path": str(out["program"]) if status == "ok" else None,
        "split": {"seed": 0, "train": len(split["train"]), "val": len(split["val"])},
        "settings": {"auto": auto, "num_candidates": num_candidates, "num_trials": num_trials, "minibatch": minibatch,
                     "max_bootstrapped_demos": 0, "max_labeled_demos": 0},
        "best_score": getattr(optimized, "score", None),
        "peak": {"trial": best_full["trial"] if best_full else None},
        "total_eval_calls": max((t["total_eval_calls_so_far"] or 0 for t in trials), default=None),
        "trials": trials, "candidate_programs": [
            {"score": c.get("score"), "instruction": da.program_instruction(c["program"])}
            for c in getattr(optimized, "candidate_programs", [])],
        "demos": da.program_demos(optimized), "elapsed_sec": round(time.time() - t0, 1), "cost": client.summary(),
    }
    out["log"].parent.mkdir(parents=True, exist_ok=True)
    out["log"].write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if status == "ok":
        out["prompt"].parent.mkdir(parents=True, exist_ok=True)
        out["prompt"].write_text(da.program_instruction(optimized) + "\n", encoding="utf-8")
        optimized.save(str(out["program"]))
    return log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mock", "execute"], default="mock")
    ap.add_argument("--max-cost", type=float)
    ap.add_argument("--start-instruction", type=Path, default=DEFAULT_START)
    ap.add_argument("--estimate", action="store_true", help="print the analytic call/token estimate, no calls")
    ap.add_argument("--prompts-dir", type=Path, default=None,
                    help="directory for prompt outputs (default: $CCD_PROMPTS_DIR or prompts/)")
    ap.add_argument("--reports-dir", type=Path, default=None,
                    help="directory for report outputs (default: $CCD_REPORTS_DIR or reports/)")
    a = ap.parse_args(argv)
    if a.estimate:
        import apo_estimate
        apo_estimate.print_table(apo_estimate.mipro_rows(label="C3-M"))
        return
    if a.mode == "execute" and a.max_cost is None:
        ap.error("--mode execute requires --max-cost")
    if w := ac.c2_freeze_warning(a.start_instruction):
        print(w)
    client = LLMClient(mode=a.mode, max_cost_usd=a.max_cost)
    out = output_paths(a.mode, a.prompts_dir, a.reports_dir)
    log = run_mipro(client, start_instruction=a.start_instruction, out=out)
    print(json.dumps({k: log[k] for k in ("status", "best_score", "total_eval_calls", "final_instruction_path")}))
    from run import print_summary
    print_summary(client.summary())
    if log["status"] != "ok":
        raise SystemExit(f"MIPROv2 run did not finish cleanly: {log['status']} {log['errors'][:3]}")


if __name__ == "__main__":
    main()
