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
[ -f data/noq_reviews.parquet ] && [ -f data/nocq_reviews.parquet ] || scripts/inputs.sh || exit 1

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

( p1 256 data/sample_reviews.parquet
  p2 160 data/panels.jsonl &
  p2 32 data/panels_placebo.jsonl &
  p2 32 data/panels_placebo_area.jsonl &
  p2 64 data/panels_sub3.jsonl &
  wait
  echo "CHAIN main DONE $(date)" ) > "$LOG_DIR/main.log" 2>&1 &

( p1 96 data/other_pass1.parquet --tag _other --paper_kind general
  p2 96 data/panels_other.jsonl --tag _other --paper_kind general &
  p2 48 data/panels_other_placebo.jsonl --tag _other --paper_kind general &
  wait
  echo "CHAIN other DONE $(date)" ) > "$LOG_DIR/other.log" 2>&1 &

( p1 128 data/noq_reviews.parquet --tag _noq
  p2 128 data/panels_noq.jsonl --tag _noq
  echo "CHAIN noq DONE $(date)" ) > "$LOG_DIR/noq.log" 2>&1 &

( p1 96 data/nocq_reviews.parquet --tag _nocq
  p2 96 data/panels_nocq.jsonl --tag _nocq
  echo "CHAIN nocq DONE $(date)" ) > "$LOG_DIR/nocq.log" 2>&1 &

( p1 64 data/icml_sample.parquet --tag _icml
  p2 64 data/panels_icml.jsonl --tag _icml
  echo "CHAIN icml DONE $(date)" ) > "$LOG_DIR/icml.log" 2>&1 &

( p1 48 data/wd_reviews.parquet --tag _wd
  p2 48 data/panels_wd.jsonl --tag _wd
  echo "CHAIN small DONE $(date)" ) > "$LOG_DIR/small.log" 2>&1 &

wait
uv run python src/extract.py merge --model "$MODEL"

if [ -f "data/overlap/panel_outcomes_$MODEL.parquet" ]; then
  RW=data/rewrite_test
  ( uv run python src/ai.py rewrite_sample --model "$MODEL"
    uv run python src/ai.py rewrite_panels
    uv run python src/ai.py rewrite_inputs --model "$MODEL"
    uv run python src/ai.py rewrite_pairs
    p1 64 $RW/reviews_ml.parquet --tag _ml --data $RW
    p1 64 $RW/reviews_general.parquet --tag _general --paper_kind general --data $RW
    p2 64 $RW/panels_ml.jsonl --tag _ml --data $RW &
    p2 64 $RW/panels_general.jsonl --tag _general --paper_kind general --data $RW &
    p2 64 $RW/panels_pairs_ml.jsonl --tag _pairs_ml --data $RW &
    p2 64 $RW/panels_pairs_general.jsonl --tag _pairs_general --paper_kind general --data $RW &
    wait
    uv run python src/extract.py merge --model "$MODEL" --data $RW
    uv run python src/extract.py pass2 $RW/panels_audit.jsonl --model opus-5.5 --claims_file "claims_$MODEL.jsonl" \
      --data $RW --threads 12 --budget 30
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
