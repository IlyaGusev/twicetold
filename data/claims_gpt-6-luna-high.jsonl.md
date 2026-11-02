# data/claims_gpt-6-luna-high.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for every review sample.
One line per review: 221,516 rows, 1,593,207 claims. Each `review_id` occurs once.

| Field | Meaning |
|---|---|
| `review_id` | Review id; see the sample table for its form |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; present in 1,509 rows only |
| `claims` | List of claims; each claim has `claim` (text), `category` (17 values), `target` (8 values), `specific` (bool) |

No row has an `error` field. 4,207 reviews have an empty claim list.

| Sample | Pass-1 input | `--tag` | `review_id` | Rows | Claims | Empty | With `effort` |
|---|---|---|---|---|---|---|---|
| Main ML | `sample_reviews.parquet` | none | OpenReview note id | 116,847 | 803,207 | 1,333 | 0 |
| ICML | `icml_sample.parquet` | `_icml` | OpenReview note id | 11,500 | 74,693 | 168 | 0 |
| No questions | `noq_reviews.parquet` | `_noq` | `noq_` + OpenReview note id | 46,972 | 242,400 | 556 | 153 |
| Journals | `other_pass1.parquet` | `_other` | journal review id | 40,992 | 426,977 | 2,063 | 1,356 |
| Withdrawn | `wd_reviews.parquet` | `_wd` | OpenReview note id | 3,891 | 29,773 | 3 | 0 |
| None | none | none | journal review id | 1,314 | 16,157 | 84 | 0 |

Each review of the five inputs has one row. The 1,314 rows of the last line are journal reviews (SciPost, F1000, BMC
Medicine, PLOS ONE, Copernicus) that are not in `other_pass1.parquet`; no panel uses them.

## Make

```
uv run python -I src/extract.py pass1 data/sample_reviews.parquet --model gpt-6-luna-high --threads 256 --budget 300
uv run python -I src/extract.py pass1 data/other_pass1.parquet --tag _other --paper_kind general --model gpt-6-luna-high --threads 96 --budget 300
uv run python -I src/extract.py pass1 data/noq_reviews.parquet --tag _noq --model gpt-6-luna-high --threads 128 --budget 300
uv run python -I src/extract.py pass1 data/icml_sample.parquet --tag _icml --model gpt-6-luna-high --threads 64 --budget 300
uv run python -I src/extract.py pass1 data/wd_reviews.parquet --tag _wd --model gpt-6-luna-high --threads 48 --budget 300
uv run python -I src/extract.py merge --model gpt-6-luna-high
```

`scripts/extract_all.sh gpt-6-luna-high` runs these. The pass-1 steps are the first steps of its `main`, `other`,
`noq`, `icml` and `small` chains, which run in parallel; `merge` runs after all chains end. Each review, cut at
14,000 characters, goes to OpenRouter with the prompt `src/prompts/pass1.j2` and a strict JSON schema. The prompt
names the paper as "machine-learning conference paper"; with `--paper_kind general` (journals) it names it as
"scientific paper submitted to a journal". It calls a paid API and needs `OPENROUTER_API_KEY`.

Without `--tag`, pass 1 appends to this file. With `--tag`, it appends to the part file `claims_gpt-6-luna-high<tag>.jsonl`.
Pass 1 skips ids that have a row without `error` in the part file or in this file. `merge` joins this file and the part
files into this file, then deletes the part files. Each `review_id` keeps the last row without `error`, else the last
error row (`extract.merge_rows`). All files are in the `--data` folder (default `data/`). To make a new copy, add
`--data <new folder>` to each command.

## Notes

- Model output; a rerun gives different claims.
- `extract.py` reads a row without `effort` as effort `high`.
- Git LFS stores it.

## Read by

- `src/outcomes.py` `load_outputs` (`load_claims` keeps only `specific`), in `panels` and `pairs`.
- `src/between.py claims_table`, which writes `data/claims_table_gpt-6-luna-high.parquet`. It skips reviews that
  are not in `reviews.parquet`, so it keeps only main and withdrawn rows.
- `src/extract.py pass2`: claims of each panel's reviews, from this file and the part file of the same tag.
- `src/extract.py pass1` (ids to skip) and `merge`.
