#!/bin/sh
while kill -0 2383282 2>/dev/null; do sleep 60; done
cd /home/yallen/twicetold
export MODEL=gpt-6-luna THREADS=64 BUDGET_USD=60
uv run python -I src/extract.py pass2 data/panels.jsonl
uv run python -I src/extract.py pass2 data/panels_placebo.jsonl
uv run python -I src/extract.py pass2 data/panels_blind.jsonl
