#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  echo "usage: $0 OUT_DIR [STEP_BUDGET_USD=15] [SAMPLER=sample|sample_audit|sample_written]" >&2
  exit 1
fi
OUT=$1
BUDGET=${2:-15}
SAMPLER=${3:-sample}
if [ "$(realpath -m "$OUT")" = "$(realpath data/gold)" ]; then
  echo "refusing to write into data/gold; pick a new OUT_DIR" >&2
  exit 1
fi
export OMP_NUM_THREADS=2
step() {
  echo "=== $* $(date +%H:%M:%S)"
  local exit_code=0
  nice -n 19 uv run --all-extras python -I src/gold.py "$@" 2>&1 | { grep --line-buffered -v "it/s\]" || true; } || exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo "FAILED (exit $exit_code): $*"
    exit "$exit_code"
  fi
}

step "$SAMPLER" --out "$OUT"
step council_claims --out "$OUT" --budget "$BUDGET"
step claim_inputs --out "$OUT"
step judge_claims --out "$OUT" --budget "$BUDGET"
step council_groups --out "$OUT" --budget "$BUDGET"
step group_inputs --out "$OUT"
step judge_groups --out "$OUT" --budget "$BUDGET"
if [ "$SAMPLER" != sample ]; then
  step model_groups --out "$OUT"
  step match_claims --gold "$OUT" --folder "$OUT/match" --budget "$BUDGET"
fi
echo "GOLD DONE $(date)"
