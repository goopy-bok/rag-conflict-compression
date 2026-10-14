#!/usr/bin/env bash
# Progress of a G1-lite run started with scripts/run_detached.sh (read-only).
cd "$(dirname "$0")/.."
pid=$(cat logs/lite.pid 2>/dev/null || true)
if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
    echo "running: pid $pid"
else
    echo "not running; last exit: $(cat logs/lite.exit 2>/dev/null || echo '-') (code time phases)"
fi
latest=$(ls -t logs/lite_*.log 2>/dev/null | head -1)
if [[ -n $latest ]]; then
    echo "phase log: $latest"
    tail -4 "$latest" | cut -c1-200
fi
echo "gates:"
for f in results/preflight.json results/g0_local/agreement.json results/pretest_gate.json; do
    [[ -f $f ]] && echo "  $f pass=$(uv run python -c "import json; print(json.load(open('$f')).get('pass'))" 2>/dev/null)"
done
[[ -f results/FROZEN_PROMPTS.json ]] && echo "  results/FROZEN_PROMPTS.json present (prompts frozen)"
echo "outputs (lines):"
ls reports/apo_log_C3-G*_seed0.json 2>/dev/null | sed 's/^/  /'
for f in results/*_r0.3_s0.jsonl results/*_r0.3_s0.jsonl.partial; do
    [[ -f $f ]] && printf "  %6d %s\n" "$(wc -l < "$f")" "$f"
done
echo "memory: $(sysctl -n vm.swapusage)"
