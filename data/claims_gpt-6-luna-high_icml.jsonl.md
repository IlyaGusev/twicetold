# data/claims_gpt-6-luna-high_icml.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 11,500 ICML reviews
of `icml_sample.parquet`. One line per review: 11,500 rows, 74,693 claims.

| Field | Meaning |
|---|---|
| `review_id` | OpenReview note id |
| `model` | Short name, always `gpt-6-luna-high` |
| `claims` | List of claims; each claim has `claim`, `category`, `target`, `specific` |

168 reviews have an empty claim list. No row has an `error` or `effort` field.

## Make

```
uv run python -I src/extract.py pass1 data/icml_sample.parquet --tag _icml --model gpt-6-luna-high --threads 64 --budget 300
```

This is the first step of the `icml` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each review, cut at 14,000
characters, goes to OpenRouter with the prompt `src/prompts/pass1.j2` and a strict JSON schema. It calls a paid API
and needs `OPENROUTER_API_KEY`. `extract.py` appends to the output in the `--data` folder (default `data/`) and skips
done ids. To make a new copy, add `--data <new folder>`.

## Notes

- Model output; a rerun gives different claims.
- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

- `src/outcomes.py` `load_claims`, in `source_panels` (tag `_icml`) and `pairs`.
- `src/extract.py pass2`, as input to pass 2.
