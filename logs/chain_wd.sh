#!/bin/sh
cd /home/yallen/twicetold
set -a; . ./.env; set +a
while pgrep -f "pass1 data/wd_reviews" >/dev/null; do sleep 30; done
MODEL=gpt-6-luna TAG=_wd THREADS=32 CLAIMS_FILES=claims_gpt-6-luna_wd.jsonl uv run python -I src/extract.py pass2 data/panels_wd.jsonl
echo WD DONE
