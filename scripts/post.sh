#!/bin/bash
# Post-extraction chain for one model: panel outcomes, pair table, standardized rates, validation reports, tables
# and figures. Without OUT_DIR the tables and figures overwrite the paper's (main model). With OUT_DIR they go to
# OUT_DIR/tables and OUT_DIR/figs, and the model-named reports still land in reports/.
# Usage: scripts/post.sh MODEL [OUT_DIR]
set -eu
cd "$(dirname "$0")/.." || exit 1
M=$1; OUT=${2:-}
export OMP_NUM_THREADS=2
R() { echo "=== $* $(date +%H:%M:%S)"; nice -n 19 uv run python -I "$@" 2>&1 | grep -v "it/s\]"; }

for t in "" _noq _wd _icml _other; do R src/outcomes.py panels --model "$M" --tag "$t"; done
R src/outcomes.py pairs --model "$M"
R src/outcomes.py standardize --model "$M"
R src/outcomes.py withdrawals --model "$M"
R src/between.py claims_table --model "$M"
R src/validate.py detector_placebo --model "$M"
R src/validate.py rq1_evidence --model "$M"
R src/validate.py variants --model "$M"
R src/validate.py blind --model "$M"
R src/validate.py reviewer_fe --model "$M"
R src/validate.py instruments
if [ -z "$OUT" ]; then
  R src/validate.py robust --model "$M"
  R src/figures.py table --model "$M"
  R src/figures.py all --model "$M"
else
  mkdir -p "$OUT/figs" "$OUT/tables"
  R src/validate.py robust --model "$M" --tex "$OUT/tables/robust.tex"
  R src/figures.py table --model "$M" --tables "$OUT/tables"
  R src/figures.py all --model "$M" --figs "$OUT/figs"
fi
echo "POST DONE $(date)"
