"""
C3-G / C3-G1: GEPA over the compressor instruction (execution guide v2 section 7.2).

  uv run python src/apo_gepa.py --condition C3-G --seed 0 --mode mock
  uv run python src/apo_gepa.py --condition C3-G1 --seed 1 --mode execute --max-cost 20
  uv run python src/apo_gepa.py --condition C3-G --seed 0 --estimate          # no calls

C3-G  : start = --start-instruction (default prompts/c2_best.txt) + fixed c2_fewshot.json examples
C3-G1 : start = prompts/c1_plain.txt, no examples
Program: dspy_adapter.BudgetedCompressor (harness prompt format, one budget retry in forward).
Metric : apo_common.judge_and_score as dspy.Prediction(score, feedback).
LMs    : task = ClientLM(compressor role, or --task-model-role aux_compressor), reflection =
         ClientLM(reflection role, temperature 1.0, max_tokens 4096). All calls go through LLMClient.
Split  : apo_common.load_apo_split(0) for every seed (the seed varies GEPA only).
Early stop: gepa stop callback, stop after --patience (5) consecutive full-val evaluations that do
         not raise the best val score; max_metric_calls is the hard cap.
Outputs: execute -> prompts/<stem>_seed{k}.txt (+ .json program), reports/apo_log_<COND>_seed{k}.json;
         mock -> the same names under prompts/mock/ and reports/mock/.
"""
from __future__ import annotations

import argparse
import json
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

CONDITIONS = {  # condition -> (output stem, default start instruction, use few-shot)
    "C3-G": ("c3g", ROOT_DIR / "prompts" / "c2_best.txt", True),
    "C3-G1": ("c3g1", ROOT_DIR / "prompts" / "c1_plain.txt", False),
}
REFLECTION_MAX_TOKENS = 4096


def output_paths(condition: str, seed: int, mode: str, task_model_role: str = "compressor") -> dict[str, Path]:
    stem = CONDITIONS[condition][0] + ("_llama" if task_model_role == "aux_compressor" else "")
    tag = condition + ("-llama" if task_model_role == "aux_compressor" else "")
    sub = "mock" if mode == "mock" else ""
    pdir, rdir = ROOT_DIR / "prompts" / sub, ROOT_DIR / "reports" / sub
    return {"prompt": pdir / f"{stem}_seed{seed}.txt", "program": pdir / f"{stem}_seed{seed}.json",
            "log": rdir / f"apo_log_{tag}_seed{seed}.json"}


class Tracker:
    """gepa callback + stopper: records every full-val evaluation and stops after `patience`
    consecutive val evaluations without a new best, or when the client reports an error."""

    def __init__(self, patience: int | None, lms: list[da.ClientLM]):
        self.patience, self.lms = patience, lms
        self.val_evals: list[dict] = []
        self.minibatch: list[dict] = []
        self.best, self.since_best, self.calls, self.stop_reason = float("-inf"), 0, 0, None

    def on_budget_updated(self, e):
        self.calls = e["metric_calls_used"]

    def on_valset_evaluated(self, e):
        score = float(e["average_score"])
        self.val_evals.append({"iteration": e["iteration"], "candidate_idx": e["candidate_idx"], "val_score": score,
                               "metric_calls_so_far": self.calls})
        if score > self.best:
            self.best, self.since_best = score, 0
        else:
            self.since_best += 1

    def on_candidate_accepted(self, e):
        self.minibatch.append({"iteration": e["iteration"], "accepted": True, "new_sum": e["new_score"]})

    def on_candidate_rejected(self, e):
        self.minibatch.append({"iteration": e["iteration"], "accepted": False,
                               "old_sum": e["old_score"], "new_sum": e["new_score"]})

    def __call__(self, state) -> bool:
        if any(lm.errors for lm in self.lms):
            self.stop_reason = "client_error"
        elif self.patience is not None and self.since_best >= self.patience:
            self.stop_reason = f"no val improvement in {self.patience} consecutive val evaluations"
        return self.stop_reason is not None


def run_gepa(condition: str, seed: int, client, *, max_metric_calls: int = 6000,
             start_instruction: Path | None = None, task_model_role: str = "compressor",
             split: dict | None = None, patience: int | None = 5, out: dict[str, Path] | None = None,
             reflection_minibatch_size: int = 3) -> dict:
    stem, default_start, use_fewshot = CONDITIONS[condition]
    start_path = Path(start_instruction or default_start)
    roles = load_models_config()["roles"]
    split = split or ac.load_apo_split(0)
    out = out or output_paths(condition, seed, client.mode, task_model_role)
    examples_block = ac.expected_examples_block(use_fewshot)
    instances = {inst["id"]: inst for part in ("train", "val") for inst, _ in split[part]}

    if not roles.get(task_model_role):
        raise SystemExit(f"role {task_model_role!r} has no model in configs/models.yaml")
    task_lm = da.ClientLM(client, roles[task_model_role], temperature=0.0)
    refl_lm = da.ClientLM(client, roles["reflection"], temperature=1.0, max_tokens=REFLECTION_MAX_TOKENS)
    tracker = Tracker(patience, [task_lm, refl_lm])
    gepa = dspy.GEPA(metric=da.make_metric(instances, client, roles["opt_judge"], feedback=True),
                     reflection_lm=refl_lm, max_metric_calls=max_metric_calls, seed=seed, track_stats=True,
                     num_threads=client.concurrency, reflection_minibatch_size=reflection_minibatch_size,
                     gepa_kwargs={"stop_callbacks": [tracker], "callbacks": [tracker]})
    ac.reset_examples_seen()
    t0 = time.time()
    with dspy.context(lm=task_lm):
        optimized = gepa.compile(da.BudgetedCompressor(start_path.read_text(encoding="utf-8"), examples_block),
                                 trainset=da.to_examples(split["train"]), valset=da.to_examples(split["val"]))
    elapsed = time.time() - t0
    errors = task_lm.errors + refl_lm.errors
    budget_hit = client.max_cost_usd is not None and client.usd_spent >= client.max_cost_usd > 0  # local runs: cap 0, never hit
    status = "error" if errors else ("budget_exceeded" if budget_hit else "ok")
    if tracker.stop_reason is None:
        tracker.stop_reason = "max_metric_calls reached"

    r = optimized.detailed_results
    best = int(r.best_idx)
    cand = [{"idx": i, "instruction": da.program_instruction(c), "val_score": float(r.val_aggregate_scores[i]),
             "parents": list(r.parents[i] or []), "metric_calls_at_discovery": int(r.discovery_eval_counts[i])}
            for i, c in enumerate(r.candidates)]
    peak = next((v for v in tracker.val_evals if v["candidate_idx"] == best), None)
    log = {
        "condition": condition, "seed": seed, "mode": client.mode, "status": status, "errors": errors[:20],
        "task_model": roles[task_model_role], "reflection_model": roles["reflection"], "judge_model": roles["opt_judge"],
        "start_instruction_path": str(start_path), "start_instruction_sha256": ac.sha256_text(start_path.read_text(encoding="utf-8")),
        "c2_freeze_warning": ac.c2_freeze_warning(start_path) if use_fewshot else None,
        "uses_fewshot": use_fewshot, "examples_sha256_expected": ac.sha256_text(examples_block),
        "examples_sha256_seen": ac.examples_seen(),
        "final_instruction_path": str(out["prompt"]) if status == "ok" else None,
        "program_path": str(out["program"]) if status == "ok" else None,
        "split": {"seed": 0, "train": len(split["train"]), "val": len(split["val"])},
        "settings": {"max_metric_calls": max_metric_calls, "patience": patience,
                     "reflection_minibatch_size": reflection_minibatch_size, "use_merge": True,
                     "candidate_selection": "pareto"},
        "stop_reason": tracker.stop_reason, "total_metric_calls": int(r.total_metric_calls or tracker.calls),
        "num_full_val_evals": r.num_full_val_evals, "best_idx": best, "best_val_score": cand[best]["val_score"],
        "peak": {"val_eval_index": tracker.val_evals.index(peak) if peak else None,
                 "iteration": peak["iteration"] if peak else None,
                 "metric_calls": cand[best]["metric_calls_at_discovery"]},
        "val_curve": tracker.val_evals, "train_minibatch": tracker.minibatch, "candidates": cand,
        "demos": da.program_demos(optimized), "elapsed_sec": round(elapsed, 1), "cost": client.summary(),
    }
    out["log"].parent.mkdir(parents=True, exist_ok=True)
    out["log"].write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    if status == "ok":
        out["prompt"].parent.mkdir(parents=True, exist_ok=True)
        out["prompt"].write_text(da.program_instruction(optimized) + "\n", encoding="utf-8")
        optimized.save(str(out["program"]))
    return log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["mock", "execute"], default="mock")
    ap.add_argument("--max-cost", type=float)
    ap.add_argument("--max-metric-calls", type=int, default=6000)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--start-instruction", type=Path)
    ap.add_argument("--task-model-role", choices=["compressor", "aux_compressor"], default="compressor")
    ap.add_argument("--estimate", action="store_true", help="print the analytic call/token estimate, no calls")
    a = ap.parse_args(argv)

    if a.estimate:
        import apo_estimate
        apo_estimate.print_table(apo_estimate.gepa_rows(a.condition, a.max_metric_calls, a.task_model_role,
                                                        label=f"{a.condition} seed{a.seed}"))
        return
    if a.mode == "execute" and a.max_cost is None:
        ap.error("--mode execute requires --max-cost")
    start = a.start_instruction or CONDITIONS[a.condition][1]
    if CONDITIONS[a.condition][2] and (w := ac.c2_freeze_warning(start)):
        print(w)
    client = LLMClient(mode=a.mode, max_cost_usd=a.max_cost)
    log = run_gepa(a.condition, a.seed, client, max_metric_calls=a.max_metric_calls, start_instruction=start,
                   task_model_role=a.task_model_role, patience=a.patience)
    print(json.dumps({k: log[k] for k in ("status", "stop_reason", "total_metric_calls", "best_idx",
                                          "best_val_score", "final_instruction_path")}, ensure_ascii=False))
    from run import print_summary
    print_summary(client.summary())
    if log["status"] != "ok":
        raise SystemExit(f"GEPA run did not finish cleanly: {log['status']} {log['errors'][:3]}")


if __name__ == "__main__":
    main()
