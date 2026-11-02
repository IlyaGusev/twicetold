# data/clusters_gpt-6-luna-high.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the main ML panels and
their controls. One line per panel: 47,979 rows, 1,047,327 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | Panel id: a paper id, `placebo_…`, `placeboA_…`, `sub3_…` or `B00`–`B39` |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; present in 8,499 rows only |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` (review 2 of `review_ids`, claim 5) |

No row has an `error` field. No panel id occurs twice.

| Panel file | Rows |
|---|---|
| `panels.jsonl` | 30,930 |
| `panels_placebo.jsonl` | 4,500 |
| `panels_placebo_area.jsonl` | 2,996 |
| `panels_sub3.jsonl` | 9,513 |
| none (ids `B00`–`B39`) | 40 |

## Make

```
uv run python -I src/extract.py pass2 data/panels.jsonl --model gpt-6-luna-high --threads 160 --budget 300
uv run python -I src/extract.py pass2 data/panels_placebo.jsonl --model gpt-6-luna-high --threads 32 --budget 300
uv run python -I src/extract.py pass2 data/panels_placebo_area.jsonl --model gpt-6-luna-high --threads 32 --budget 300
uv run python -I src/extract.py pass2 data/panels_sub3.jsonl --model gpt-6-luna-high --threads 64 --budget 300
```

These are the pass-2 steps of the `main` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each panel's claims from
`claims_gpt-6-luna-high.jsonl` go to OpenRouter as `R<review>-<claim> [category; target]: text` lines, with the
prompt `src/prompts/pass2.j2` and a strict JSON schema. It calls a paid API and needs `OPENROUTER_API_KEY`. Pass 2
reads the claims file from the `--data` folder (default `data/`), appends to the output there and skips done panels.
To make a new copy, add `--data <new folder>` and put a claims file there first. No command makes the 40 `B…` rows.

## Notes

- Model output; a rerun gives different groups.
- 1,516 `sub3_…` rows list the panel's reviews in another order than `panels_sub3.jsonl`. Readers use the order in
  the row, so the claim ids stay correct.
- 572 rows leave out 605 claims, 146 rows put 179 claims in two or more groups, and 19 rows have 22 ids that point to
  no claim. `metrics.assign` puts a left-out claim in its own cluster, keeps the first cluster of a repeated claim and ignores ids that point to no claim.
- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

`src/outcomes.py` `ml_panels` (tag `""`) and `pairs`, through `metrics.load_jsonl` (skips rows with `error`; for a
repeated panel id the last row wins) and `metrics.assign`. Both skip the `B…` rows, because these ids are not papers.
