# data/clusters_gpt-6-luna-high_other.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the journal panels:
`panels_other.jsonl` (17,520 same-paper panels) and `panels_other_placebo.jsonl` (6,450 control panels). 23,971 rows
for 23,970 panel ids, 582,050 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | `other_<paper id>` or `otherplacebo_<source>_<year>_<n>` |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; present in 7,131 rows only |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` (absent in the 135 rows with no claims) |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` |
| `error` | Error text, in place of `targets` and `clusters`, when every attempt failed |

135 rows have `clusters: []` and no `targets`: no review of these panels has a claim, so `extract.py` made no call.
5 rows are error rows (`JSONDecodeError('Expecting value: line 1 column 1 (char 0)')`). These 4 panels have no
usable row:

| Panel | Error rows |
|---|---|
| `other_2112.03919v1` | 2 |
| `otherplacebo_F1000_2022_117` | 1 |
| `otherplacebo_F1000_2022_31` | 1 |
| `otherplacebo_PLOS_2025_1` | 1 |

Each of the other 23,966 panels has one usable row, with the same `review_ids` in the same order as the panel file.

## Make

```
uv run python -I src/extract.py pass2 data/panels_other.jsonl --tag _other --paper_kind general --model gpt-6-luna-high --threads 96 --budget 300
uv run python -I src/extract.py pass2 data/panels_other_placebo.jsonl --tag _other --paper_kind general --model gpt-6-luna-high --threads 48 --budget 300
```

These are the pass-2 steps of the `other` chain in `scripts/extract_all.sh gpt-6-luna-high`. Each panel's claims
from `claims_gpt-6-luna-high_other.jsonl` go to OpenRouter with the prompt `src/prompts/pass2.j2` (journal wording)
and a strict JSON schema. It calls a paid API and needs `OPENROUTER_API_KEY`. Pass 2 reads the claims file from the
`--data` folder (default `data/`), appends to the output there and skips panels with a usable row. To make a new
copy, add `--data <new folder>` and put a claims file there first.

## Notes

- Model output; a rerun gives different groups.
- 442 usable rows leave out 579 claims, 148 rows put 224 claims in two or more groups, and 29 rows have 50 ids that
  point to no claim. `metrics.assign` puts a left-out claim in its own cluster, keeps the first cluster of a repeated claim and ignores ids that point to no claim.
- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

`src/outcomes.py` `source_panels` (tag `_other`) and `pairs`, through `metrics.load_jsonl` and `metrics.assign`.
`load_jsonl` skips rows with `error`, so the 4 failed panels are not in the outcomes; for a repeated panel id the last
usable row wins. Both readers skip a row whose `review_ids` differ from the current panel file.
