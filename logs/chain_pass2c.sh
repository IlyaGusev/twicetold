#!/bin/sh
cd /home/yallen/twicetold
set -a; . ./.env; set +a
export MODEL=gpt-6-luna THREADS=192 BUDGET_USD=70
uv run python -I src/extract.py pass2 data/panels.jsonl
uv run python -I src/extract.py pass2 data/panels_placebo.jsonl
uv run python -I src/extract.py pass2 data/panels_blind.jsonl
uv run python -I src/extract.py pass2 data/panels_placebo_area.jsonl
uv run python -I src/extract.py pass2 data/panels_sub3.jsonl
TAG=_noq CLAIMS_FILES=claims_gpt-6-luna_noq.jsonl uv run python -I src/extract.py pass2 data/panels_noq.jsonl
TAG=_llm CLAIMS_FILES=claims_gpt-6-luna_llm.jsonl,claims_gpt-6-luna.jsonl uv run python -I src/extract.py pass2 data/panels_llm.jsonl
echo CHAIN DONE
