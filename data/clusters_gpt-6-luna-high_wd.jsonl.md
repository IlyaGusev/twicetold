# data/clusters_gpt-6-luna-high_wd.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 1,000 panels of
withdrawn papers in `panels_wd.jsonl`. One line per panel: 1,000 rows, 24,610 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | `wd_<paper id>` |
| `model` | Short name, always `gpt-6-luna-high` |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` |

No row has an `error` or `effort` field. No panel id occurs twice. Each panel of `panels_wd.jsonl` has one row, with
the same `review_ids` in the same order.

## Make

```
uv run python -I src/extract.py pass2 data/panels_wd.jsonl --tag _wd --model gpt-6-luna-high --threads 48 --budget 300
```

This is the pass-2 step of the `small` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each panel's claims from
`claims_gpt-6-luna-high_wd.jsonl` go to OpenRouter with the prompt `src/prompts/pass2.j2` and a strict JSON schema.
It calls a paid API and needs `OPENROUTER_API_KEY`. Pass 2 reads the claims file from the `--data` folder (default
`data/`), appends to the output there and skips done panels. To make a new copy, add `--data <new folder>` and put a
claims file there first.

## Notes

- Model output; a rerun gives different groups.
- 20 rows leave out 21 claims and 1 row has 1 id that points to no claim. No row puts a claim in two groups. `metrics.assign` puts a left-out claim in its own cluster, keeps the first cluster of a repeated claim and ignores ids that point to no claim.
- Git LFS stores it.

## Read by

`src/outcomes.py` `ml_panels` (tag `_wd`), through `metrics.load_jsonl` (skips rows with `error`; for a repeated
panel id the last row wins) and `metrics.assign`.
