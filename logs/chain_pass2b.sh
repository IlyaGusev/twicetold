#!/bin/sh
while kill -0 2387203 2>/dev/null || kill -0 2387206 2>/dev/null; do sleep 60; done
cd /home/yallen/twicetold
set -a; . ./.env; set +a
export MODEL=gpt-6-luna THREADS=64 BUDGET_USD=30
uv run python -I src/extract.py pass2 data/panels_placebo_area.jsonl
uv run python -I src/extract.py pass2 data/panels_sub3.jsonl
TAG=_noq CLAIMS_FILES=claims_gpt-6-luna_noq.jsonl uv run python -I src/extract.py pass2 data/panels_noq.jsonl
TAG=_llm CLAIMS_FILES=claims_gpt-6-luna_llm.jsonl,claims_gpt-6-luna.jsonl uv run python -I src/extract.py pass2 data/panels_llm.jsonl
echo CHAIN2 DONE
