# data/claims_gpt-6-luna-high_other.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the journal reviews of
`other_pass1.parquet` (SciPost, F1000, BMC Medicine, PLOS ONE, Copernicus). One line per review: 42,306 rows,
443,134 claims.

| Field | Meaning |
|---|---|
| `review_id` | Review id from the journal source |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; present in 1,356 rows only |
| `claims` | List of claims; each claim has `claim`, `category`, `target`, `specific` |

2,147 reviews have an empty claim list. No row has an `error` field. Each of the 40,992 reviews of
`other_pass1.parquet` has one row. The other 1,314 rows belong to reviews that are not in `other_pass1.parquet`; no
panel uses them.

## Make

```
uv run python -I src/extract.py pass1 data/other_pass1.parquet --tag _other --paper_kind general --model gpt-6-luna-high --threads 96 --budget 300
```

This is the first step of the `other` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each review, cut at 14,000
characters, goes to OpenRouter with the prompt `src/prompts/pass1.j2` (journal wording) and a strict JSON schema. It
calls a paid API and needs `OPENROUTER_API_KEY`. `extract.py` appends to the output in the `--data` folder (default
`data/`) and skips done ids. To make a new copy, add `--data <new folder>`.

## Notes

- Model output; a rerun gives different claims.
- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

- `src/outcomes.py` `load_claims`, in `source_panels` (tag `_other`) and `pairs`.
- `src/extract.py pass2`, as input to pass 2.
