#!/usr/bin/env bash
# Start scripts/run_lite.sh detached from the terminal and from any agent session (nohup), e.g.
#   scripts/run_detached.sh all
#   PREFLIGHT_ALLOW_SWAP=1 scripts/run_detached.sh all      # env overrides pass through
# Writes logs/detached_<ts>.out, logs/lite.pid while running, logs/lite.exit at the end, and posts a macOS
# notification with the exit code. Progress: scripts/lite_status.sh. Stop: kill "$(cat logs/lite.pid)"
# (run_lite.sh stops the Splash server it started on exit).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if [[ -f logs/lite.pid ]] && kill -0 "$(cat logs/lite.pid)" 2>/dev/null; then
    echo "already running: pid $(cat logs/lite.pid) (scripts/lite_status.sh)"
    exit 1
fi
[[ $# -ge 1 ]] || { echo "usage: $0 <run_lite.sh phases...|all>"; exit 2; }
OUT="logs/detached_$(date +%Y%m%d_%H%M%S).out"
rm -f logs/lite.exit
nohup bash -c '
    scripts/run_lite.sh "$@"
    rc=$?
    echo "$rc $(date "+%Y-%m-%d %H:%M:%S") $*" > logs/lite.exit
    osascript -e "display notification \"run_lite $* ended, exit $rc\" with title \"ccd-g1 G1-lite\"" >/dev/null 2>&1 || true
    rm -f logs/lite.pid
    exit "$rc"' _ "$@" >"$OUT" 2>&1 &
echo $! > logs/lite.pid
echo "started pid $(cat logs/lite.pid): run_lite.sh $*"
echo "output $OUT; progress: scripts/lite_status.sh"
