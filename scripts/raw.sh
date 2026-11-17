#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2
step() {
  echo "=== $* $(date +%H:%M:%S)"
  nice -n 19 uv run python "$@" 2>&1 | { grep --line-buffered -v "it/s\]" || true; }
}

step src/corpus.py other
step src/fetch.py icml --year 2025
step src/fetch.py icml --year 2026
step src/panels.py icml
step src/panels.py other
step src/panels.py robust
step src/panels.py nocq
step src/meld.py input
[ -f data/raw/meld_scores.jsonl ] || { echo "run src/meld.py scores on a GPU, then rerun"; exit 1; }
step src/meld.py flags
[ -f data/raw/editlens_scores.jsonl ] || { echo "run src/editlens.py scores on a GPU, then rerun"; exit 1; }
step src/editlens.py flags
[ -f data/raw/gpttrace_scores.jsonl ] || { echo "run src/gpttrace.py scores on a GPU, then rerun"; exit 1; }
step src/gpttrace.py flags
[ -f data/raw/gigacheck_scores.jsonl ] || { echo "run src/gigacheck_detector.py scores on a GPU, then rerun"; exit 1; }
step src/gigacheck_detector.py flags
step src/abstracts.py input
for detector in meld editlens gigacheck; do
  [ -f "data/raw/${detector}_abstract_scores.jsonl" ] || { echo "score data/raw/abstract_input.parquet with $detector on a GPU, then rerun"; exit 1; }
done
step src/abstracts.py flags
step src/release.py from_raw
echo "RAW DONE $(date)"
