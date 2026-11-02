# data/gold/claims_opus-5.5_tune.jsonl

Council claims of `opus-5.5` (OpenRouter `anthropic/claude-opus-5.5`) for the 240 reviews of `tune_reviews.parquet`. One line per
review: `review_id`, `model`, `claims` (each claim: `claim`, `category`, `target`, `specific`). 1,809 claims.

## Make

```
uv run python -I src/gold.py council_claims --out <dir>
```

Step 2 of `scripts/gold.sh <dir>`. It runs `src/extract.py pass1` with tag `_tune` for each model of
`settings.GOLD_COUNCIL`, two times, so the second run retries failed reviews. Reasoning effort medium; review text cut
at `settings.REVIEW_TEXT_MAX_CHARS`. Paid API; needs network and the OpenRouter key.

## Notes

- Model output; a rerun gives different claims.
- Rows have no `effort` field. Files from the current code have one.

## Read by

`src/gold.py claim_inputs`.
