#!/usr/bin/env bash
# G1-lite (2-day reduced run, preregistration section 7; results are PRELIMINARY, not the formal G1 decision).
#
# Usage (from anywhere; the script cd's to the ccd-g1 root):
#   scripts/run_lite.sh preflight   # amendment 2.2-c: roles, G0 input, Ollama/thinking, Splash, disk/swap, seal, tests
#   scripts/run_lite.sh g0auto      # amendment 2.2-d: label sheet -> rejudge opt/eval -> negative control ->
#                                   # label-free checks (inter-judge kappa, negative-control FP) -> g0_auto.json
#   scripts/run_lite.sh gepa        # needs G0-auto pass; GEPA seed 0, 400 metric calls: C3-G1 then C3-G
#   scripts/run_lite.sh gate        # amendment 2.2-c pre-test gate (src/pretest_gate.py); freeze/test need its pass
#   scripts/run_lite.sh freeze      # test_id subset (pool first 200) + freeze C0 C1 C2 C3-G1 C3-G (seed 0)
#   scripts/run_lite.sh test        # 5 prompts x {test_id 200 (+downstream, D), ood1 100, ood2 154, distractor 100}
#   scripts/run_lite.sh stats       # reports/<date>_g1_lite_results.md (no LLM calls; regenerated every run)
#   scripts/run_lite.sh all         # preflight g0auto gepa gate freeze test stats, stopping where a gate is not met
#   scripts/run_lite.sh g0 stats    # LAST, by the author (2.2-d): accuracy vs author labels -> agreement.json;
#                                   # label with: uv run python src/label_server.py (do not open label_key.json)
# Unattended: scripts/run_detached.sh all (nohup, notification at the end); progress: scripts/lite_status.sh
# Before g0: configs/models.yaml roles fixed with `uv run python src/configure_roles.py --compressor <id>`, and
# $PILOT_DIR/<compressor>_{C0,C1}.jsonl present (PILOT_DIR default results/pilot_h2h: the amendment 2.2-b
# head-to-head outputs; set PILOT_DIR=results/pilot_select for the original 5-candidate pilot).
# Overrides (set only by a person; recorded in the JSON outputs):
#   PREFLIGHT_ALLOW_SWAP=1        preflight P5 swap check (amendment 2.2-c)
#   GATE_ALLOW_NO_IMPROVEMENT=1   gate K3: C3-G1 did not beat its C1 start on val (2.2-c)
#   GATE_ACCEPT_K5=C2,C3-G        gate K5 failure accepted per prompt; read by src/pretest_gate.py (2.2-g)
# Other environment:
#   CCD_CACHE_DIR=<dir>           LLM cache directory, default cache/ (the 2.2-h rerun used results/rerun_2.2-h_cache)
#   PILOT_DIR=<dir>               G0 input pilot, see above
#
# Every phase skips work whose output file already exists (run.py also resumes its .partial files), so a
# phase can be rerun after an interruption. All LLM calls are local (Ollama, Splash); every run uses
# `--mode execute --max-cost 0`. Each command runs under `caffeinate -i`. If a phase needs the Splash model
# (eval_judge or reflection role), Splash is started on 127.0.0.1:8010 before and stopped after the phase
# (port 8000 is never touched). Logs: logs/lite_<phase>_<timestamp>.log.
#
# Expected time (concurrency 1; per-call times assumed: compression 10-15 s, judge / downstream 5-20 s,
# Splash 27B judge slower; the pilot measured 29 s wall per compression because of swap and sleep):
#   preflight: seconds (+ unit tests)
#   g0     : 200 judge calls (2 judges x (50 rejudge + 50 negative))          ~0.5-1.5 h  (+ author labelling 1-2 h)
#   gepa   : 2 x 400 metric calls (compress + judge) + reflection calls        ~5-8 h
#   gate   : 5 prompts x 100 dev compressions (no judge; amendment 2.2-e)     ~1-1.5 h
#   freeze : seconds
#   test   : test_id 5x200x4 calls = 4,000; others 5x354x2 = 3,540 calls       ~20-35 h
#   stats  : seconds
set -euo pipefail

cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y-%m-%d)
SPLASH_MODEL="incoai/Qwen3.8-27B-Splash"
SPLASH_PORT=8010
SPLASH_URL="http://127.0.0.1:${SPLASH_PORT}/v1/models"
SPLASH_WAIT_S=${SPLASH_WAIT_S:-900}
G0=results/g0_local
PILOT_DIR=${PILOT_DIR:-results/pilot_h2h}
export PILOT_DIR
GATE=results/pretest_gate.json
R=0.3
CONDS=(C0 C1 C2 C3-G1 C3-G)
EXE=(--mode execute --max-cost 0)
SPLASH_PID=""
LOG=""
mkdir -p logs

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# run CMD...: under caffeinate -i, output appended to the phase log
run() {
    log "+ $*"
    caffeinate -i "$@" 2>&1 | tee -a "$LOG"
}

role() {  # resolved model id of a role in configs/models.yaml
    uv run python -c "import sys; sys.path.insert(0, 'src'); from llm_client import load_models_config as c; print(c()['roles'].get('$1') or '')"
}

splash_up() {
    if curl -sf -m 5 "$SPLASH_URL" >/dev/null 2>&1; then
        [[ -n "$SPLASH_PID" ]] || log "Splash already answers on port $SPLASH_PORT; reusing it (not stopped by this script)"
        return 0
    fi
    log "starting Splash ($SPLASH_MODEL) on 127.0.0.1:$SPLASH_PORT"
    splash serve --model "$SPLASH_MODEL" --host 127.0.0.1 --port "$SPLASH_PORT" --no-webui \
        --max-memory 24G --default-reasoning-effort none >>"$LOG" 2>&1 &
    SPLASH_PID=$!
    caffeinate -i -w "$SPLASH_PID" &   # keep awake while the server lives; exits with it
    local waited=0
    until curl -sf -m 5 "$SPLASH_URL" >/dev/null 2>&1; do
        if ! kill -0 "$SPLASH_PID" 2>/dev/null; then
            SPLASH_PID=""
            log "Splash exited before it was ready; see $LOG"
            return 1
        fi
        if (( waited >= SPLASH_WAIT_S )); then
            log "Splash not ready after ${SPLASH_WAIT_S}s"
            splash_down
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    # /v1/models may answer before the weights finish loading: send one tiny completion and
    # wait for it, so the first real judge call does not absorb the load time or fail.
    if ! curl -sf -m 900 "http://127.0.0.1:${SPLASH_PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SPLASH_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"temperature\":0}" \
        >/dev/null 2>&1; then
        log "Splash warm-up request failed; see $LOG"
        splash_down
        return 1
    fi
    log "Splash ready after ${waited}s plus warm-up (pid $SPLASH_PID)"
}

splash_down() {  # stops only a server this script started
    [[ -n "$SPLASH_PID" ]] || return 0
    log "stopping Splash (pid $SPLASH_PID)"
    kill "$SPLASH_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SPLASH_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SPLASH_PID" 2>/dev/null || true
    wait "$SPLASH_PID" 2>/dev/null || true
    SPLASH_PID=""
    if curl -sf -m 5 "$SPLASH_URL" >/dev/null 2>&1; then
        log "warning: port $SPLASH_PORT still answers after stopping pid; check with: lsof -nP -iTCP:$SPLASH_PORT -sTCP:LISTEN"
    fi
}
trap splash_down EXIT

splash_if() {  # start Splash when any given role is the Splash model
    local r
    for r in "$@"; do
        if [[ "$(role "$r")" == "$SPLASH_MODEL" ]]; then splash_up; return; fi
    done
}

g0_passed() {
    [[ -f "$G0/agreement.json" ]] && uv run python -c \
        "import json, sys; sys.exit(0 if json.load(open('$G0/agreement.json'))['pass'] else 1)"
}

g0auto_passed() {
    [[ -f "$G0/g0_auto.json" ]] && uv run python -c \
        "import json, sys; sys.exit(0 if json.load(open('$G0/g0_auto.json'))['pass'] else 1)"
}

require_g0auto() {  # amendment 2.2-d: GEPA/gate/freeze/test need the label-free G0 checks; the author part runs last
    if ! g0auto_passed; then
        log "G0-auto not passed ($G0/g0_auto.json missing or pass=false). Stopping before $1 (amendment 2.2-d)."
        exit 1
    fi
}

gate_passed() {
    [[ -f "$GATE" ]] && uv run python -c \
        "import json, sys; sys.exit(0 if json.load(open('$GATE'))['pass'] else 1)"
}

require_gate() {  # amendment 2.2-c: the sealed test is opened only after the pre-test gate passed
    if ! gate_passed; then
        log "pre-test gate not passed ($GATE missing or pass=false). Stopping before $1 (amendment 2.2-c)."
        exit 1
    fi
}

# ------------------------------------------------------------------------------------------------ phases

g0_prepare() {  # label sheet, negative-control items, both judges' rejudge and negative-control runs (skips done work)
    local comp stem vj=(uv run python src/validate_judge.py)
    comp=$(role compressor)
    stem=${comp//:/-}
    local res=("$PILOT_DIR/${stem}_C0.jsonl" "$PILOT_DIR/${stem}_C1.jsonl")
    for f in "${res[@]}"; do [[ -f "$f" ]] || { log "missing pilot output $f (compressor role = $comp)"; exit 1; }; done
    local samp=(--results "${res[@]}" --n 50 --out-dir "$G0")
    [[ -f "$G0/author_labels.csv" ]] || run "${vj[@]}" label-sheet "${samp[@]}"
    [[ -f "$G0/negative_control_items.jsonl" ]] || run "${vj[@]}" negative-control "${samp[@]}"
    local j
    for j in opt eval; do
        if [[ ! -f "$G0/rejudge_$j.jsonl" || ! -f "$G0/negative_control_$j.jsonl" ]]; then
            splash_if "${j}_judge"
        fi
        [[ -f "$G0/rejudge_$j.jsonl" ]] || run "${vj[@]}" rejudge --judge "$j" "${samp[@]}" "${EXE[@]}"
        [[ -f "$G0/negative_control_$j.jsonl" ]] || \
            run "${vj[@]}" negative-control --run --judge "$j" --out-dir "$G0" "${EXE[@]}"
    done
    splash_down
}

phase_g0auto() {  # amendment 2.2-d: the G0 criteria that need no author labels, before GEPA
    g0_prepare
    local rc=0
    run uv run python src/validate_judge.py judge-checks --dir "$G0" || rc=$?
    if (( rc != 0 )); then
        log "G0-AUTO FAIL: see $G0/g0_auto.json (inter-judge kappa / negative control); later phases will not run"
        exit 1
    fi
    log "G0-AUTO PASS (accuracy vs the author is checked at the end: uv run python src/label_server.py, then $0 g0 stats)"
}

phase_g0() {  # amendment 2.2-d: the author part of G0 (accuracy vs author labels), run last
    g0_prepare
    local rc=0
    run uv run python src/validate_judge.py agreement --dir "$G0" || rc=$?
    if (( rc == 3 )); then
        log "author labels incomplete: label with 'uv run python src/label_server.py' (or fill $G0/author_labels.csv;"
        log "do not open $G0/label_key.json), then rerun: $0 g0 stats"
        return 3
    fi
    (( rc == 0 )) || exit "$rc"
    g0_passed && log "G0 PASS" || log "G0 FAIL: see $G0/agreement.json and amendment 2.2-d section 4 (what follows a late failure)"
}

phase_gepa() {
    require_g0auto gepa
    local cond stem
    for cond in C3-G1 C3-G; do
        stem=$([[ $cond == C3-G1 ]] && echo c3g1 || echo c3g)
        if [[ -f "prompts/${stem}_seed0.txt" ]]; then log "skip $cond: prompts/${stem}_seed0.txt exists"; continue; fi
        splash_if reflection compressor opt_judge
        run uv run python src/apo_gepa.py --condition "$cond" --seed 0 --max-metric-calls 400 "${EXE[@]}"
    done
    splash_down
}

phase_preflight() {
    local args=()
    [[ ${PREFLIGHT_ALLOW_SWAP:-0} == 1 ]] && args+=(--allow-swap)
    local rc=0
    run uv run python src/preflight.py ${args[@]+"${args[@]}"} || rc=$?
    if (( rc != 0 )); then
        log "PREFLIGHT FAIL: see results/preflight.json; nothing else was started."
        exit "$rc"
    fi
}

phase_gate() {
    require_g0auto gate
    if gate_passed; then log "skip gate: $GATE already passed"; return 0; fi   # resume after the test was opened
    local args=()
    [[ ${GATE_ALLOW_NO_IMPROVEMENT:-0} == 1 ]] && args+=(--allow-no-improvement)
    local rc=0
    run uv run python src/pretest_gate.py ${args[@]+"${args[@]}"} || rc=$?
    if (( rc != 0 )); then
        log "PRE-TEST GATE FAIL: see $GATE. The sealed test stays closed; later phases will not run."
        exit "$rc"
    fi
    log "PRE-TEST GATE PASS"
}

phase_freeze() {
    require_g0auto freeze
    require_gate freeze
    run uv run python src/make_test_subset.py --split test_id --n 200
    local fz=results/FROZEN_PROMPTS.json
    if [[ -f $fz ]] && grep -q 'prompts/c3g_seed0.txt' $fz && grep -q 'prompts/c3g1_seed0.txt' $fz; then
        log "skip freeze: $fz already lists the C3 seed-0 prompts"
        return 0
    fi
    run uv run python src/run.py freeze --conditions "${CONDS[@]}" --seeds 0
}

phase_test() {
    require_g0auto test
    require_gate test
    [[ -f results/FROZEN_PROMPTS.json ]] || { log "run the freeze phase first"; exit 1; }
    local sp c extra out conds
    for sp in test_id test_ood1 test_ood2 distractor; do
        case $sp in
            test_id) extra=() ;;
            test_ood1|distractor) extra=(--limit 100 --no-downstream) ;;
            test_ood2) extra=(--no-downstream) ;;
        esac
        # Preregistration 7.1 (OOD reduction, 2026-10-09): H4 needs only C3-G on OOD; C1 kept as reference.
        case $sp in
            test_ood1|test_ood2) conds=(C1 C3-G) ;;
            *) conds=("${CONDS[@]}") ;;
        esac
        for c in "${conds[@]}"; do
            out="results/${c}_${sp}_r${R}_s0.jsonl"
            if [[ -f $out ]]; then log "skip $out (exists)"; continue; fi
            splash_if eval_judge compressor downstream
            run uv run python src/run.py run --condition "$c" --split "$sp" --r "$R" --seed 0 "${EXE[@]}" ${extra[@]+"${extra[@]}"}
        done
    done
    splash_down
}

phase_stats() {
    run uv run python src/stats.py lite --out "reports/${DATE}_g1_lite_results.md"
}

# -------------------------------------------------------------------------------------------------- main

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; }

[[ $# -ge 1 ]] || { usage; exit 2; }
phases=("$@")
[[ "$1" == all ]] && phases=(preflight g0auto gepa gate freeze test stats)
for p in "${phases[@]}"; do
    case $p in
        preflight|g0auto|g0|gepa|gate|freeze|test|stats) ;;
        *) usage; exit 2 ;;
    esac
done
for p in "${phases[@]}"; do
    LOG="logs/lite_${p}_${TS}.log"
    log "== phase $p =="
    log "ollama $(curl -s -m 5 http://localhost:11434/api/version || echo unreachable)"  # a mid-run update would show here
    "phase_$p"   # plain call so that set -e stays active inside the phase; a failure ends the script
done
if [[ "$1" == all ]]; then
    log "ALL AUTOMATIC PHASES DONE. Remaining for the author (amendment 2.2-d): label the 50 G0 items with"
    log "  uv run python src/label_server.py   (opens http://127.0.0.1:8765), then: $0 g0 stats"
fi
