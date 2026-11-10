#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 OUT_DIR [STEP_BUDGET_USD=15]" >&2
  exit 1
fi
OUT=$1
BUDGET=${2:-15}
if [ "$(realpath -m "$OUT")" = "$(realpath data/gold)" ]; then
  echo "refusing to write into data/gold; pick a new OUT_DIR" >&2
  exit 1
fi
export OMP_NUM_THREADS=2
step() {
  echo "=== $* $(date +%H:%M:%S)"
  local exit_code=0
  nice -n 19 uv run --all-extras python src/gold.py "$@" 2>&1 | { grep --line-buffered -v "it/s\]" || true; } || exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo "FAILED (exit $exit_code): $*"
    exit "$exit_code"
  fi
}

step sample --out "$OUT"
step model_claims --out "$OUT" --budget "$BUDGET"
step council_claims --out "$OUT" --budget "$BUDGET"
step model_groups --out "$OUT" --budget "$BUDGET"
step council_groups --out "$OUT" --budget "$BUDGET"
step member_groups --out "$OUT" --budget "$BUDGET"
step kim_sample --out "$OUT"
step external_claims --out "$OUT" --name kim --paper_kind general --budget "$BUDGET"
step claimcheck_sample --out "$OUT"
step external_claims --out "$OUT" --name claimcheck --paper_kind ml --budget "$BUDGET"
echo "GOLD DONE $(date)"
