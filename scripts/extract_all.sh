#!/bin/bash
set -u
cd "$(dirname "$0")/.." || exit 1
PASS_BUDGET_USD=${2:-300}
if [ $# -lt 1 ] || [ $# -gt 2 ] || ! [[ $PASS_BUDGET_USD =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "usage: $0 MODEL [PASS_BUDGET_USD=300]" >&2
  exit 1
fi
MODEL=$1
LOG_DIR=logs/extract_$MODEL
mkdir -p "$LOG_DIR"
[ -d data/final ] || uv run python src/release.py pull || exit 1

run_pass() {
  local pass=$1 threads=$2
  shift 2
  uv run python src/extract.py "$pass" "$@" \
    --model "$MODEL" --threads "$threads" --budget "$PASS_BUDGET_USD" 2>&1 |
    grep --line-buffered -v "it/s\]"
  local exit_code=${PIPESTATUS[0]}
  if [ "$exit_code" -ne 0 ]; then
    echo "PASS FAILED (exit $exit_code): $pass $*"
  fi
}
p1() { run_pass pass1 "$@"; }
p2() { run_pass pass2 "$@"; }

( run_pass final_pass1 256
  run_pass final_pass2 160
  uv run python src/release.py build --model "$MODEL"
  echo "CHAIN final DONE $(date)" ) > "$LOG_DIR/final.log" 2>&1

if [ -f "data/work/overlap/panel_outcomes_$MODEL.parquet" ]; then
  RW=data/additional/rewrite_test
  ( uv run python src/ai.py rewrite_sample --model "$MODEL"
    uv run python src/ai.py rewrite_panels
    uv run python src/ai.py rewrite_inputs --model "$MODEL"
    uv run python src/ai.py rewrite_pairs
    p1 64 $RW/reviews_ml.parquet --data $RW
    p1 64 $RW/reviews_general.parquet --paper_kind general --data $RW
    p2 64 $RW/panels_ml.jsonl --data $RW
    p2 64 $RW/panels_general.jsonl --paper_kind general --data $RW
    p2 64 $RW/panels_pairs_ml.jsonl --data $RW
    p2 64 $RW/panels_pairs_general.jsonl --paper_kind general --data $RW
    echo "CHAIN rewrite DONE $(date)" ) > "$LOG_DIR/rewrite.log" 2>&1
  uv run python src/ai.py polish > "$LOG_DIR/polish.log" 2>&1
  uv run python src/fetch.py arxiv_papers > "$LOG_DIR/arxiv.log" 2>&1
  uv run python src/ai.py paper_reviews > "$LOG_DIR/paper_reviews.log" 2>&1
fi
echo "ALL DONE $(date)"
if grep -l "Traceback\|budget cap\|PASS FAILED\|panels skipped\|items failed" "$LOG_DIR"/*.log; then
  echo "SOME CHAIN FAILED"
  exit 1
fi
