# data/gold/claims_sol-or_tune.jsonl

Council claims of `sol-or` (OpenRouter `openai/gpt-6.1-sol`) for the 240 reviews of `tune_reviews.parquet`. One line per review:
`review_id`, `model`, `claims` (each claim: `claim`, `category`, `target`, `specific`). 1,976 claims.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

`src/extract.py pass1 data/tune/tune_reviews.parquet --model sol-or --tag _tune --effort medium --threads 16 --budget 15`
on 2026-10-31 at 14:57 UTC. The prompt is the pass-1 prompt for ML papers; it is identical to the current
`src/prompts/pass1.j2`. Strict JSON schema, reasoning effort medium, review text cut at 14,000 characters. All 240
calls succeeded; cost about $1.89. The `extract.py` of that time did not write an `effort` field, so rows have
none. The current code (`src/gold.py council_claims`) writes it.

## Reproduce

`src/gold.py council_claims --out <dir>` (or `scripts/gold.sh <dir>`). Model output is not deterministic, so a new
run gives other claims.

## Read by

`src/gold.py claim_inputs`.
