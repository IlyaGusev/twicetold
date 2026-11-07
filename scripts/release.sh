#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ $# -ne 1 ]; then
  echo "usage: $0 MODEL" >&2
  exit 1
fi
MODEL=$1
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

step src/extract.py merge --model "$MODEL"
step src/meld.py flags
step src/release.py build --model "$MODEL"
step src/release.py artifacts
echo "RELEASE BUILT $(date)"
