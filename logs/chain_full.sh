#!/bin/sh
cd /home/yallen/twicetold
set -a; . ./.env; set +a
export MODEL=gpt-6-luna THREADS=160 BUDGET_USD=45
TAG=_noq uv run python -I src/extract.py pass1 data/noq_full_reviews.parquet
uv run python -I src/extract.py pass2 data/panels_sub3_full.jsonl
TAG=_noq CLAIMS_FILES=claims_gpt-6-luna_noq.jsonl uv run python -I src/extract.py pass2 data/panels_noq_full.jsonl
echo FULL DONE
