#!/bin/sh
cd /home/yallen/twicetold
until grep -q "FULL DONE" logs/chain_full.log; do sleep 60; done
uv run python -I src/outcomes.py gpt-6-luna
uv run python -I src/outcomes.py gpt-6-luna _noq
uv run python -I src/analysis_variants.py gpt-6-luna > reports/variants_full_gpt-6-luna.txt 2>&1
uv run python -I src/make_tables.py gpt-6-luna
echo FULL2 DONE
