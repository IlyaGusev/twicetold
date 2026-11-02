# data/gold/claims_council.jsonl

Council claims for the 240 reviews of `tune_reviews.parquet`, from each model of `settings.GOLD_COUNCIL`: `sol-or`
(OpenRouter `openai/gpt-6.1-sol`), `opus-5.5` (`anthropic/claude-opus-5.5`) and `kimi-k3.1-pro`
(`moonshotai/kimi-k3.1-pro`). 720 lines: one line per model and review, key `model` + `review_id`. Fields:
`review_id`, `model`, `claims` (each claim: `claim`, `category`, `target`, `specific`).

| Model | Lines | Claims |
|---|---|---|
| `sol-or` | 240 | 1,976 |
| `opus-5.5` | 240 | 1,809 |
| `kimi-k3.1-pro` | 240 | 1,830 |

## Make

```
uv run python -I src/gold.py council_claims --out <dir>
```

Step 2 of `scripts/gold.sh <dir>`. It runs `src/extract.py pass1` with tag `_tune` for each council model, two times,
so the second run retries failed reviews. Then it joins the part files `claims_<model>_tune.jsonl` into this file
(`extract.merge_rows`: per key, the last row without `error`, else the last error row) and deletes them. Reasoning
effort medium; review text cut at `settings.REVIEW_TEXT_MAX_CHARS`. Paid API; needs network and the OpenRouter key.

## Notes

- Model output; a rerun gives different claims.
- Rows have no `effort` field. Rows from the current code have one.
- Git LFS stores it.

## Read by

`src/gold.py claim_inputs`.
