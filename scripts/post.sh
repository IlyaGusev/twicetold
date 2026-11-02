#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 MODEL [OUT_DIR]" >&2
  exit 1
fi
MODEL=$1
OUT_DIR=${2:-}
export OMP_NUM_THREADS=2
step() {
  echo "=== $* $(date +%H:%M:%S)"
  local exit_code=0
  nice -n 19 uv run --all-extras python -I "$@" 2>&1 | { grep --line-buffered -v "it/s\]" || true; } || exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo "FAILED (exit $exit_code): $*"
    exit "$exit_code"
  fi
}

for tag in "" _noq _wd _icml _other; do
  step src/outcomes.py panels --model "$MODEL" --tag "$tag"
done
step src/outcomes.py pairs --model "$MODEL"
step src/between.py claims_table --model "$MODEL"
step src/between.py embed --model "$MODEL"
step src/validate.py detector_placebo --model "$MODEL"
step src/validate.py rq1_evidence --model "$MODEL"
if [ -z "$OUT_DIR" ]; then
  step src/validate.py robust --model "$MODEL"
  step src/figures.py table --model "$MODEL"
  step src/figures.py numbers --model "$MODEL"
  step src/figures.py all --model "$MODEL"
else
  mkdir -p "$OUT_DIR/figs" "$OUT_DIR/tables"
  step src/validate.py robust --model "$MODEL" --tex "$OUT_DIR/tables/robust.tex"
  step src/figures.py table --model "$MODEL" --tables "$OUT_DIR/tables"
  step src/figures.py numbers --model "$MODEL" --tables "$OUT_DIR/tables"
  step src/figures.py all --model "$MODEL" --figs "$OUT_DIR/figs"
fi
echo "POST DONE $(date)"
