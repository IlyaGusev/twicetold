# data/claims_gpt-6-luna-high.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 116,847 reviews of
`sample_reviews.parquet`. One line per review: 116,847 rows, 803,207 claims.

| Field | Meaning |
|---|---|
| `review_id` | OpenReview note id |
| `model` | Short name, always `gpt-6-luna-high` |
| `claims` | List of claims; each claim has `claim` (text), `category` (17 values), `target` (8 values), `specific` (bool) |

1,333 reviews have an empty claim list. No row has an `error` or `effort` field.

## Make

```
uv run python -I src/extract.py pass1 data/sample_reviews.parquet --model gpt-6-luna-high --threads 256 --budget 300
```

This is the first step of the `main` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each review, cut at 14,000
characters, goes to OpenRouter with the prompt `src/prompts/pass1.j2` and a strict JSON schema. It calls a paid API
and needs `OPENROUTER_API_KEY`. `extract.py` appends to `claims_gpt-6-luna-high.jsonl` in the `--data` folder (default
`data/`) and skips done ids. To make a new copy, add `--data <new folder>`.

## Notes

- Model output; a rerun gives different claims.
- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

- `src/outcomes.py` `load_claims`, in `ml_panels` (tag `""`) and `pairs`.
- `src/between.py claims_table`, which writes `data/claims_table_gpt-6-luna-high.parquet`. It skips reviews that
  are not in `reviews.parquet`.
- `src/extract.py pass2`, as input to pass 2.
