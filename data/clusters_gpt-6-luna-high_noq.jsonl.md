# data/clusters_gpt-6-luna-high_noq.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 12,000 panels of
`panels_noq.jsonl` (reviews without their questions field). One line per panel: 12,000 rows, 209,515 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | `noq_<paper id>` |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; present in 151 rows only |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` |

No row has an `error` field. No panel id occurs twice. Each panel of `panels_noq.jsonl` has one row, with the same
`review_ids` in the same order.

## Make

```
uv run python -I src/extract.py pass2 data/panels_noq.jsonl --tag _noq --model gpt-6-luna-high --threads 128 --budget 300
```

This is the pass-2 step of the `noq` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each panel's claims from
`claims_gpt-6-luna-high_noq.jsonl` go to OpenRouter with the prompt `src/prompts/pass2.j2` and a strict JSON schema.
It calls a paid API and needs `OPENROUTER_API_KEY`. Pass 2 reads the claims file from the `--data` folder (default
`data/`), appends to the output there and skips done panels. To make a new copy, add `--data <new folder>` and put a
claims file there first.

## Notes

- Model output; a rerun gives different groups.
- 96 rows leave out 99 claims and 23 rows put 24 claims in two groups. No id points to no claim. `metrics.assign` puts a left-out claim in its own cluster, keeps the first cluster of a repeated claim and ignores ids that point to no claim.
- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

`src/outcomes.py` `ml_panels` (tag `_noq`), through `metrics.load_jsonl` (skips rows with `error`; for a repeated
panel id the last row wins) and `metrics.assign`.
