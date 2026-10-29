#!/bin/sh
cd /home/yallen/twicetold
export MODEL=qwen3-4b LOCAL_MODEL_PATH=data/local/Qwen3-4B-Instruct-2507-Q4_K_M.gguf THREADS=1 BUDGET_USD=1
uv run python -I src/extract.py pass1 data/blind_reviews.parquet
uv run python -I src/extract.py pass2 data/panels_blind.jsonl
uv run python -I src/validate_blind.py qwen3-4b > reports/validate_blind_qwen3-4b.txt 2>&1
echo LOCAL DONE
