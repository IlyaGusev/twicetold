# data/claims_gpt-6-luna-high_noq.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 46,972 reviews of
`noq_reviews.parquet` (reviews without their questions field). One line per review: 46,972 rows, 242,400 claims.

| Field | Meaning |
|---|---|
| `review_id` | OpenReview note id |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; present in 153 rows only |
| `claims` | List of claims; each claim has `claim`, `category`, `target`, `specific` |

556 reviews have an empty claim list. No row has an `error` field.

## Make

```
uv run python -I src/extract.py pass1 data/noq_reviews.parquet --tag _noq --model gpt-6-luna-high --threads 128 --budget 300
```

This is the first step of the `noq` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each review, cut at 14,000
characters, goes to OpenRouter with the prompt `src/prompts/pass1.j2` and a strict JSON schema. It calls a paid API
and needs `OPENROUTER_API_KEY`. `extract.py` appends to the output in the `--data` folder (default `data/`) and skips
done ids. To make a new copy, add `--data <new folder>`.

## Notes

- Model output; a rerun gives different claims.
- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

- `src/outcomes.py` `load_claims`, in `ml_panels` (tag `_noq`).
- `src/extract.py pass2`, as input to pass 2.
