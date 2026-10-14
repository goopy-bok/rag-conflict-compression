"""
Expanded Sealed Benchmark Runner (Stage 3 & Stage 4 Power Expansion):
- Preserves original results/openrouter_phi4/ (N=200 test_id, N=100 ood1, N=154 ood2, N=100 distractor) untouched.
- Expands evaluation in results/openrouter_phi4_expanded/ using the exact same FROZEN_PROMPTS.json:
  * test_id:    N = 600 (0001..0600 from data/sealed/test_id_pool.jsonl; +400 unused sealed instances)
  * test_ood1:  N = 300 (0001..0300 full data/sealed/test_ood1.jsonl; +200 unused sealed instances)
  * test_ood2:  N = 154 (0001..0154 full data/sealed/test_ood2.jsonl; already complete)
  * distractor: N = 300 (0001..0300 full data/sealed/distractor.jsonl; +200 unused sealed instances)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


def seed_partials(orig_dir: Path, exp_dir: Path, conds: list[str]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(orig_dir / "FROZEN_PROMPTS.json", exp_dir / "FROZEN_PROMPTS.json")
    for c in conds:
        # test_ood2 is already full (154 instances)
        ood2_src = orig_dir / f"{c}_test_ood2_r0.3_s0.jsonl"
        ood2_dst = exp_dir / f"{c}_test_ood2_r0.3_s0.jsonl"
        if ood2_src.exists() and not ood2_dst.exists():
            shutil.copy2(ood2_src, ood2_dst)

        # For test_id (200 -> 600), test_ood1 (100 -> 300), distractor (100 -> 300),
        # seed .partial from the existing 200/100 completed rows if neither final nor partial exists yet
        for sp in ("test_id", "test_ood1", "distractor"):
            src = orig_dir / f"{c}_{sp}_r0.3_s0.jsonl"
            dst_final = exp_dir / f"{c}_{sp}_r0.3_s0.jsonl"
            dst_part = exp_dir / f"{c}_{sp}_r0.3_s0.jsonl.partial"
            if src.exists() and not dst_final.exists() and not dst_part.exists():
                shutil.copy2(src, dst_part)


def main() -> None:
    load_env(ROOT_DIR / ".env")
    os.environ["CCD_MODELS_CONFIG"] = str(ROOT_DIR / "configs/models_openrouter.yaml")
    os.environ["CCD_CONDITIONS_CONFIG"] = str(ROOT_DIR / "configs/conditions_openrouter.yaml")
    orig_dir = ROOT_DIR / "results/openrouter_phi4"
    exp_dir = ROOT_DIR / "results/openrouter_phi4_expanded"
    os.environ["CCD_RESULTS_DIR"] = str(exp_dir)
    os.environ["CCD_CACHE_DIR"] = str(ROOT_DIR / "cache_openrouter")

    conds = ["C2", "C3-M", "C3-P", "C1", "C0"]
    seed_partials(orig_dir, exp_dir, conds)

    from llm_client import LLMClient
    from run import print_summary, run_condition

    client = LLMClient(mode="execute", max_cost_usd=20.0, concurrency=24)
    t0 = time.time()

    def run_with_retry(c: str, sp: str, limit: int, data_path: Path | None, downstream: bool) -> None:
        for attempt in range(1, 6):
            try:
                run_condition(c, sp, 0.3, client, seed=0, limit=limit,
                              data_path=data_path, results_dir=exp_dir, downstream=downstream)
                return
            except Exception as e:
                if "BudgetExceeded" in type(e).__name__:
                    raise
                print(f"  [WARN] {c} {sp} attempt {attempt} transient error: {e!r}; retrying in 5s...", flush=True)
                time.sleep(5)
        raise RuntimeError(f"Failed {c} {sp} after 5 attempts")

    # Phase 1: Expand test_id (200 -> 600) across all 5 conditions (with Downstream QA + D Judge)
    for c in conds:
        out_file = exp_dir / f"{c}_test_id_r0.3_s0.jsonl"
        if out_file.exists() and sum(1 for _ in out_file.open(encoding="utf-8")) >= 600:
            print(f"[SKIP] {out_file.name} already has 600 rows")
            continue
        print(f"\n>>> Running {c} test_id (target N=600) | elapsed={time.time()-t0:.1f}s | spent=${client.usd_spent:.4f}",
              flush=True)
        run_with_retry(c, "test_id", 600, ROOT_DIR / "data/sealed/test_id_pool.jsonl", True)

    # Phase 2: Expand test_ood1 (100 -> 300) and distractor (100 -> 300) across all 5 conditions
    for sp in ("test_ood1", "distractor"):
        for c in conds:
            out_file = exp_dir / f"{c}_{sp}_r0.3_s0.jsonl"
            if out_file.exists() and sum(1 for _ in out_file.open(encoding="utf-8")) >= 300:
                print(f"[SKIP] {out_file.name} already has 300 rows")
                continue
            print(f"\n>>> Running {c} {sp} (target N=300) | elapsed={time.time()-t0:.1f}s | spent=${client.usd_spent:.4f}",
                  flush=True)
            run_with_retry(c, sp, 300, None, False)

    print_summary(client.summary())
    print(f"\nALL EXPANDED RUNS COMPLETED in {time.time()-t0:.1f}s | total_spent=${client.usd_spent:.4f}", flush=True)


if __name__ == "__main__":
    main()
