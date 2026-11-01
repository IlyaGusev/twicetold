#!/bin/bash
set -u
cd "$(dirname "$0")/.." || exit 1
MODEL=$1
PASS1_EFFORT=${2:-high}
PASS2_EFFORT=${3:-high}
BUDGET_USD=${4:-300}
LOG_DIR=logs/extract_$MODEL
mkdir -p "$LOG_DIR"

run_pass() {
  local pass=$1 effort=$2 threads=$3
  shift 3
  uv run python -I src/extract.py "$pass" "$@" \
    --model "$MODEL" --threads "$threads" --budget "$BUDGET_USD" --effort "$effort" 2>&1 |
    grep -v "it/s\]"
}
p1() { run_pass pass1 "$PASS1_EFFORT" "$@"; }
p2() { run_pass pass2 "$PASS2_EFFORT" "$@"; }

( p1 256 data/sample_reviews.parquet
  p2 160 data/panels.jsonl &
  p2 32 data/panels_placebo.jsonl &
  p2 32 data/panels_placebo_area.jsonl &
  p2 32 data/panels_sub3.jsonl &
  p2 32 data/panels_sub3_full.jsonl &
  p2 8 data/panels_blind.jsonl &
  wait
  echo "CHAIN main DONE $(date)" ) > "$LOG_DIR/main.log" 2>&1 &

( p1 96 data/other_pass1.parquet --tag _other --paper_kind general
  p2 96 data/panels_other.jsonl --tag _other --paper_kind general &
  p2 48 data/panels_other_placebo.jsonl --tag _other --paper_kind general &
  wait
  echo "CHAIN other DONE $(date)" ) > "$LOG_DIR/other.log" 2>&1 &

( p1 32 data/noq_reviews.parquet --tag _noq
  p1 96 data/noq_full_reviews.parquet --tag _noq
  p2 32 data/panels_noq.jsonl --tag _noq &
  p2 96 data/panels_noq_full.jsonl --tag _noq &
  wait
  echo "CHAIN noq DONE $(date)" ) > "$LOG_DIR/noq.log" 2>&1 &

( p1 64 data/icml_pass1.parquet --tag _icml
  p2 64 data/panels_icml.jsonl --tag _icml
  echo "CHAIN icml DONE $(date)" ) > "$LOG_DIR/icml.log" 2>&1 &

( p1 48 data/wd_reviews.parquet --tag _wd
  p2 48 data/panels_wd.jsonl --tag _wd
  p1 48 data/llm_reviews.parquet --tag _llm
  p2 48 data/panels_llm.jsonl --tag _llm --claims_files "claims_${MODEL}_llm.jsonl,claims_${MODEL}.jsonl"
  p1 32 data/ampere_reviews.parquet --tag _ampere
  echo "CHAIN small DONE $(date)" ) > "$LOG_DIR/small.log" 2>&1 &

wait
echo "ALL DONE $(date)"
if grep -l "Traceback\|budget cap" "$LOG_DIR"/*.log; then
  echo "SOME CHAIN FAILED"
fi
exit 0
