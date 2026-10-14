#!/usr/bin/env bash
# Step 4 of the OpenRouter Phi-4 runbook (2026-10-14): 5 conditions x 4 test splits at r = 0.3, seed 0.
# test_ood1 / distractor use the first 100 instances (--limit 100), as G1-lite did, for 1:1 comparability.
# --max-cost per run = TOTAL_CAP - cumulative spend in $CCD_RESULTS_DIR/api_cost_log.jsonl, so the cap holds
# across runs (LLMClient's own cap counts from 0 per process). A failed run is retried after 30 s (run.py
# resumes from its .partial file); BudgetExceeded stops everything. Finished outputs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_openrouter.sh >/dev/null
TOTAL_CAP=${TOTAL_CAP:-30.0}
LOGDIR=logs/openrouter
mkdir -p "$LOGDIR"

spent() {
  python3 -c "import json,sys; print(sum(json.loads(l).get('usd') or 0 for l in open(sys.argv[1])))" \
    "$CCD_RESULTS_DIR/api_cost_log.jsonl" 2>/dev/null || echo 0
}

for cond in C0 C1 C2 C3-P C3-M; do
  for split in test_id test_ood1 test_ood2 distractor; do
    out="$CCD_RESULTS_DIR/${cond}_${split}_r0.3_s0.jsonl"
    [ -f "$out" ] && { echo "SKIP $cond $split (exists)"; continue; }
    case "$split" in
      test_id) extra=() ;;
      test_ood2) extra=(--no-downstream) ;;
      *) extra=(--limit 100 --no-downstream) ;;
    esac
    log="$LOGDIR/step4_${cond}_${split}.log"
    for attempt in 1 2 3 4 5; do
      left=$(python3 -c "print(round($TOTAL_CAP - $(spent), 4))")
      echo "RUN $cond $split attempt=$attempt max_cost=$left"
      if python3 -c "import sys; sys.exit(0 if $left > 0.05 else 1)"; then :; else
        echo "STOP budget exhausted (spent $(spent))"; exit 2; fi
      uv run python src/run.py run --condition "$cond" --split "$split" --r 0.3 --seed 0 \
        --mode execute --max-cost "$left" "${extra[@]}" >>"$log" 2>&1
      rc=$?
      [ $rc -eq 0 ] && [ -f "$out" ] && break
      if grep -q "BudgetExceeded" "$log"; then echo "STOP BudgetExceeded in $log (spent $(spent))"; exit 2; fi
      echo "FAIL $cond $split rc=$rc; retry in 30 s"; sleep 30
    done
    [ -f "$out" ] || { echo "ERROR: $out was not created"; exit 1; }
    echo "OK $cond $split n=$(wc -l <"$out" | tr -d ' ') cumulative_usd=$(spent)"
  done
done
echo "ALL DONE files=$(ls "$CCD_RESULTS_DIR"/*_r0.3_s0.jsonl | wc -l | tr -d ' ') partial=$(ls "$CCD_RESULTS_DIR"/*.partial 2>/dev/null | wc -l | tr -d ' ')"
