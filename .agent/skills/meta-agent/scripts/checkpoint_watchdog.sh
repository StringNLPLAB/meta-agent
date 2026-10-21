#!/usr/bin/env bash
# Record review-debt transitions; pass the selected log, run and cadence as arguments.
set -uo pipefail
: "${EAU_ROOT:?Set EAU_ROOT to the training checkout}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="${PYTHON_BIN:-python3}"
LOG="$EAU_ROOT/reports/checkpoint_debt.log"
mkdir -p "$EAU_ROOT/reports"
prev=""
while true; do
    out=$("$PY" "$SCRIPT_DIR/pending_checkpoints.py" "$@" 2>&1)
    status=$?
    if [ "$status" -gt 1 ]; then
        echo "$out" >&2
        exit "$status"
    fi
    if [ "$out" != "$prev" ]; then
        printf '%s  %s\n' "$(date '+%F %T')" "$out" >> "$LOG"
        prev="$out"
    fi
    case "$out" in *TEARDOWN*) exit 0;; esac
    sleep "${WATCHDOG_INTERVAL:-600}"
done
