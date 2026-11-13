#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ "${1:-}" = "--council-agreement" ]; then
  shift
  OMP_NUM_THREADS=2 nice -n 19 uv run python src/validate.py council_agreement "$@"
  exit $?
fi
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
  nice -n 19 uv run --all-extras python "$@" 2>&1 | { grep --line-buffered -v "it/s\]" || true; } || exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo "FAILED (exit $exit_code): $*"
    exit "$exit_code"
  fi
}

[ -d data/final ] || step src/release.py pull
step src/outcomes.py panels --model "$MODEL"
step src/outcomes.py pairs --model "$MODEL"
step src/between.py claims_table --model "$MODEL"
step src/between.py embed --model "$MODEL"
step src/validate.py detector_placebo --model "$MODEL"
step src/validate.py rq1_evidence --model "$MODEL"
step src/validate.py journal_gaps --model "$MODEL"
step src/validate.py pair_types --model "$MODEL"
step src/validate.py pair_type_sizes --model "$MODEL"
step src/validate.py repro --model "$MODEL"
step src/validate.py quotes --model "$MODEL"
step src/validate.py external --name kim --model "$MODEL"
step src/validate.py external --name claimcheck --model "$MODEL"
if [ -z "$OUT_DIR" ]; then
  step src/validate.py robust --model "$MODEL"
  step src/validate.py forms --model "$MODEL"
  step src/validate.py extraction --model "$MODEL"
  step src/validate.py council_agreement --model "$MODEL"
  step src/validate.py pipelines --model "$MODEL"
  for check in identification journal_subsets coverage decomposition rewrite_test detector_choices second_detector; do
    step src/validate.py "$check" --model "$MODEL"
  done
  step src/measures.py measures --model "$MODEL"
  step src/figures.py table --model "$MODEL"
  step src/figures.py numbers --model "$MODEL"
  step src/figures.py prompts
  step src/figures.py all --model "$MODEL"
else
  mkdir -p "$OUT_DIR/figs" "$OUT_DIR/tables"
  step src/validate.py robust --model "$MODEL" --tex "$OUT_DIR/tables/robust.tex"
  step src/validate.py forms --model "$MODEL" --tex "$OUT_DIR/tables/forms.tex"
  step src/validate.py extraction --model "$MODEL" --tex "$OUT_DIR/tables/extraction.tex"
  step src/validate.py council_agreement --model "$MODEL" --tex "$OUT_DIR/tables/council.tex"
  step src/validate.py pipelines --model "$MODEL" --contrasts_tex "$OUT_DIR/tables/pipeline_contrasts.tex"
  for check in identification journal_subsets coverage decomposition rewrite_test detector_choices second_detector; do
    step src/validate.py "$check" --model "$MODEL" --tex "$OUT_DIR/tables/$check.tex"
  done
  step src/measures.py measures --model "$MODEL" --tex "$OUT_DIR/tables/measures.tex"
  step src/figures.py table --model "$MODEL" --tables "$OUT_DIR/tables"
  step src/figures.py numbers --model "$MODEL" --tables "$OUT_DIR/tables"
  step src/figures.py all --model "$MODEL" --figs "$OUT_DIR/figs"
fi
echo "POST DONE $(date)"
