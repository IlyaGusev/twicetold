# data/clusters_gpt-6-luna-high_icml.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 3,000 ICML panels
of `panels_icml.jsonl`. One line per panel: 3,000 rows, 65,070 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | `icml_<paper id>` |
| `model` | Short name, always `gpt-6-luna-high` |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` |

No row has an `error` or `effort` field. No panel id occurs twice. Each panel of `panels_icml.jsonl` has one row, with
the same `review_ids` in the same order.

## Make

```
uv run python -I src/extract.py pass2 data/panels_icml.jsonl --tag _icml --model gpt-6-luna-high --threads 64 --budget 300
```

This is the pass-2 step of the `icml` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each panel's claims from
`claims_gpt-6-luna-high_icml.jsonl` go to OpenRouter with the prompt `src/prompts/pass2.j2` and a strict JSON
schema. It calls a paid API and needs `OPENROUTER_API_KEY`. Pass 2 reads the claims file from the `--data` folder
(default `data/`), appends to the output there and skips done panels. To make a new copy, add `--data <new folder>`
and put a claims file there first.

## Notes

- Model output; a rerun gives different groups.
- 39 rows leave out 39 claims, 11 rows put 11 claims in two groups, and 1 row has 1 id that points to no claim.
  `metrics.assign` puts a left-out claim in its own cluster, keeps the first cluster of a repeated claim and ignores ids that point to no claim.
- Git LFS stores it.

## Read by

`src/outcomes.py` `source_panels` (tag `_icml`) and `pairs`, through `metrics.load_jsonl` (skips rows with `error`;
for a repeated panel id the last row wins) and `metrics.assign`. Both skip a row whose `review_ids` differ from the
current panel file.
