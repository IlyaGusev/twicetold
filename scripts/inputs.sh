#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=2
step() {
  echo "=== $* $(date +%H:%M:%S)"
  nice -n 19 uv run python -I "$@" 2>&1 | { grep --line-buffered -v "it/s\]" || true; }
}

step src/corpus.py other
step src/fetch.py icml --year 2025
step src/fetch.py icml --year 2026
step src/panels.py icml
step src/panels.py other
step src/panels.py robust
step src/panels.py nocq
step src/meld.py input
echo "INPUTS DONE $(date)"
