# data/clusters_gpt-6-luna-high.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for every panel file.
One line per panel: 91,738 rows, 1,997,678 clusters. Each `panel_id` occurs once.

| Field | Meaning |
|---|---|
| `panel_id` | Panel id; its prefix gives the sample (see the table below) |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; present in 19,567 rows only |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` (absent in error rows and in the 135 rows with no claims) |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` (review 2 of `review_ids`, claim 5) |
| `error` | Error text, in place of `targets` and `clusters`, when every attempt failed |

| Panel file | `panel_id` | `--tag` | Rows | Clusters | With `effort` | Error rows |
|---|---|---|---|---|---|---|
| `panels.jsonl` | paper id | none | 30,930 | 697,232 | 3 | 0 |
| `panels_placebo.jsonl` | `placebo_…` | none | 4,500 | 116,352 | 0 | 0 |
| `panels_placebo_area.jsonl` | `placeboA_…` | none | 2,996 | 58,262 | 1,200 | 0 |
| `panels_sub3.jsonl` | `sub3_<paper id>` | none | 9,513 | 174,441 | 7,296 | 0 |
| `panels_icml.jsonl` | `icml_<paper id>` | `_icml` | 3,000 | 65,070 | 0 | 0 |
| `panels_noq.jsonl` | `noq_<paper id>` | `_noq` | 12,000 | 209,515 | 151 | 0 |
| `panels_nocq.jsonl` | `nocq_<paper id>` | `_nocq` | 3,789 | 69,066 | 3,789 | 0 |
| `panels_other.jsonl` | `other_<paper id>` | `_other` | 17,520 | 386,718 | 680 | 1 |
| `panels_other_placebo.jsonl` | `otherplacebo_<source>_<year>_<n>` | `_other` | 6,450 | 195,372 | 6,448 | 2 |
| `panels_wd.jsonl` | `wd_<paper id>` | `_wd` | 1,000 | 24,610 | 0 | 0 |
| none | `B00`–`B39` | none | 40 | 1,040 | 0 | 0 |

Each panel of each panel file has one row. Counts of clusters and `effort` cover rows without `error`. 135 journal rows
(128 `other_`, 7 `otherplacebo_`) have `clusters: []` and no `targets`: no review of these panels has a claim, so
`extract.py` made no call.

These 3 panels have only an error row:

| Panel | Error |
|---|---|
| `other_2112.03919v1` | `JSONDecodeError('Expecting value: line 1 column 1 (char 0)')` |
| `otherplacebo_F1000_2022_31` | `JSONDecodeError('Expecting value: line 1 column 1 (char 0)')` |
| `otherplacebo_F1000_2022_117` | `JSONDecodeError('Expecting value: line 1 column 1 (char 0)')` |

1,516 `sub3_…` rows list the panel's reviews in another order than `panels_sub3.jsonl`. Readers use the order in the
row, so the claim ids stay correct. Every other row has the same `review_ids` in the same order as its panel file.

## Make

```
uv run python -I src/extract.py pass2 data/panels.jsonl --model gpt-6-luna-high --threads 160 --budget 300
uv run python -I src/extract.py pass2 data/panels_placebo.jsonl --model gpt-6-luna-high --threads 32 --budget 300
uv run python -I src/extract.py pass2 data/panels_placebo_area.jsonl --model gpt-6-luna-high --threads 32 --budget 300
uv run python -I src/extract.py pass2 data/panels_sub3.jsonl --model gpt-6-luna-high --threads 64 --budget 300
uv run python -I src/extract.py pass2 data/panels_other.jsonl --tag _other --paper_kind general --model gpt-6-luna-high --threads 96 --budget 300
uv run python -I src/extract.py pass2 data/panels_other_placebo.jsonl --tag _other --paper_kind general --model gpt-6-luna-high --threads 48 --budget 300
uv run python -I src/extract.py pass2 data/panels_noq.jsonl --tag _noq --model gpt-6-luna-high --threads 128 --budget 300
uv run python -I src/extract.py pass2 data/panels_nocq.jsonl --tag _nocq --model gpt-6-luna-high --threads 96 --budget 100
uv run python -I src/extract.py pass2 data/panels_icml.jsonl --tag _icml --model gpt-6-luna-high --threads 64 --budget 300
uv run python -I src/extract.py pass2 data/panels_wd.jsonl --tag _wd --model gpt-6-luna-high --threads 48 --budget 300
uv run python -I src/extract.py merge --model gpt-6-luna-high
```

`scripts/extract_all.sh gpt-6-luna-high` runs these. The pass-2 steps follow pass 1 in its `main`, `other`, `noq`,
`nocq`, `icml` and `small` chains, which run in parallel; `merge` runs after all chains end. No command makes the 40 `B…` rows.

Pass 2 reads the claims of each panel's reviews from `claims_gpt-6-luna-high.jsonl` and the part file of the same
tag. It skips a panel if one of its reviews has no claims row. It sends the claims to OpenRouter as
`R<review>-<claim> [category; target]: text` lines, with the prompt `src/prompts/pass2.j2` and a strict JSON schema.
The prompt names the paper as "machine-learning paper"; with `--paper_kind general` (journals) it names it as
"scientific paper". It calls a paid API and needs `OPENROUTER_API_KEY`.

Without `--tag`, pass 2 appends to this file. With `--tag`, it appends to the part file
`clusters_gpt-6-luna-high<tag>.jsonl`. Pass 2 skips panels that have a row without `error` in the part file or in this
file. `merge` joins this file and the part files into this file, then deletes the part files. Each `panel_id` keeps the
last row without `error`, else the last error row (`extract.merge_rows`). All files are in the `--data` folder
(default `data/`). To make a new copy, add `--data <new folder>` to each command and put a claims file there first.

## Notes

- Model output; a rerun gives different groups.
- Some rows do not follow the prompt. `metrics.assign` puts a left-out claim in its own cluster, keeps the first
  cluster of a repeated claim and ignores ids that point to no claim.

| Panel file | Rows that leave out claims (claims) | Rows that put a claim in 2+ groups (claims) | Rows with ids that point to no claim (ids) |
|---|---|---|---|
| `panels.jsonl` | 404 (429) | 111 (133) | 12 (13) |
| `panels_placebo.jsonl` | 64 (66) | 14 (14) | 1 (3) |
| `panels_placebo_area.jsonl` | 24 (25) | 3 (3) | 1 (1) |
| `panels_sub3.jsonl` | 80 (85) | 18 (29) | 5 (5) |
| `panels_icml.jsonl` | 39 (39) | 11 (11) | 1 (1) |
| `panels_noq.jsonl` | 96 (99) | 23 (24) | 0 (0) |
| `panels_nocq.jsonl` | 24 (27) | 15 (18) | 1 (3) |
| `panels_other.jsonl` | 270 (365) | 108 (163) | 29 (50) |
| `panels_other_placebo.jsonl` | 172 (214) | 40 (57) | 0 (0) |
| `panels_wd.jsonl` | 20 (21) | 0 (0) | 1 (1) |
| none (`B…`) | 2 (2) | 1 (1) | 0 (0) |

- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

- `src/outcomes.py` `load_outputs`, through `metrics.load_jsonl` (skips rows with `error`, so the 3 failed panels are
  not in the outcomes). `tag_of` splits the rows by `panel_id` prefix (`PANEL_PREFIXES`). `panels` sends tags `_icml`
  and `_other` to `source_panels` and tags `""`, `_noq`, `_nocq` and `_wd` to `ml_panels`. `pairs` reads tags `""`, `_icml`
  and `_other`. All use `metrics.assign`. `source_panels` and `pairs` skip a journal or ICML row whose `review_ids`
  differ from the current panel file. `ml_panels` and `pairs` skip the `B…` rows, because these ids are not papers.
- `src/extract.py pass2` (panels to skip) and `merge`.
- `src/validate.py reference`: the 60 rows of the reference panels (`B…` and 20 `placebo_ICLR_…` rows).
