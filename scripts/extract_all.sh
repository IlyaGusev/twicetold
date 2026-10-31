#!/bin/bash
# Every extraction pass for one model, as five independent chains in parallel. Each pass resumes from its output
# file, so a rerun after a crash is safe. Logs: logs/extract_<model>/<chain>.log.
# Usage: scripts/extract_all.sh MODEL [PASS1_EFFORT=low] [PASS2_EFFORT=medium] [BUDGET_USD=300]
set -u
cd "$(dirname "$0")/.." || exit 1
M=$1; E1=${2:-low}; E2=${3:-medium}; B=${4:-300}
LOGS=logs/extract_$M
mkdir -p "$LOGS"
p1() { local t=$1; shift; uv run python -I src/extract.py pass1 "$@" --model "$M" --threads "$t" --budget "$B" --effort "$E1" 2>&1 | grep -v "it/s\]"; }
p2() { local t=$1; shift; uv run python -I src/extract.py pass2 "$@" --model "$M" --threads "$t" --budget "$B" --effort "$E2" 2>&1 | grep -v "it/s\]"; }

( p1 256 data/sample_reviews.parquet
  p2 160 data/panels.jsonl &
  p2 32 data/panels_placebo.jsonl &
  p2 32 data/panels_placebo_area.jsonl &
  p2 32 data/panels_sub3.jsonl &
  p2 32 data/panels_sub3_full.jsonl &
  p2 8 data/panels_blind.jsonl &
  wait; echo "CHAIN main DONE $(date)" ) > "$LOGS/main.log" 2>&1 &

( p1 96 data/other_pass1.parquet --tag _other --paper_kind general
  p2 96 data/panels_other.jsonl --tag _other --paper_kind general &
  p2 48 data/panels_other_placebo.jsonl --tag _other --paper_kind general &
  wait; echo "CHAIN other DONE $(date)" ) > "$LOGS/other.log" 2>&1 &

( p1 32 data/noq_reviews.parquet --tag _noq
  p1 96 data/noq_full_reviews.parquet --tag _noq
  p2 32 data/panels_noq.jsonl --tag _noq &
  p2 96 data/panels_noq_full.jsonl --tag _noq &
  wait; echo "CHAIN noq DONE $(date)" ) > "$LOGS/noq.log" 2>&1 &

( p1 64 data/icml_pass1.parquet --tag _icml
  p2 64 data/panels_icml.jsonl --tag _icml
  echo "CHAIN icml DONE $(date)" ) > "$LOGS/icml.log" 2>&1 &

( p1 48 data/wd_reviews.parquet --tag _wd; p2 48 data/panels_wd.jsonl --tag _wd
  p1 48 data/llm_reviews.parquet --tag _llm
  # LLM-written panels mix rewritten and original reviews, so pass 2 needs both claim files
  p2 48 data/panels_llm.jsonl --tag _llm --claims_files "claims_${M}_llm.jsonl,claims_${M}.jsonl"
  p1 32 data/ampere_reviews.parquet --tag _ampere
  echo "CHAIN small DONE $(date)" ) > "$LOGS/small.log" 2>&1 &

wait
echo "ALL DONE $(date)"
grep -l "Traceback\|budget cap" "$LOGS"/*.log && echo "SOME CHAIN FAILED"
exit 0
