# data/paper_areas.parquet

Primary area for each sampled ICLR and NeurIPS paper and each ICLR 2026 paper with an author-chosen area. 44,406
rows.

| Column | Meaning |
|---|---|
| `paper_id` | OpenReview forum id |
| `area` | Author-chosen ICLR 2026 primary area when the paper has one; otherwise `area_pred` |
| `area_prob` | Classifier probability of `area_pred` |
| `area_pred` | Area the classifier predicts from title and abstract |
| `row` | Row of the paper in `abstract_emb.npy` (equal to the row number here) |
| `area_true` | Primary area from the `MlouisBE/iclr-rebuttal-analysis` files (ICLR 2023–2026); empty otherwise |
| `area_final` | `area_true` when present; otherwise `area` |
| `area_source` | `openreview` when `area_true` is present (29,249 rows); otherwise `predicted` (15,157 rows) |

There are 21 areas. Rows per venue and year (through the current `papers.parquet`):

| Year | ICLR | NeurIPS |
|---|---|---|
| 2017 | 479 | – |
| 2018 | 918 | – |
| 2019 | 1,412 | – |
| 2020 | 2,127 | – |
| 2021 | 2,594 | 1,000 |
| 2022 | 2,617 | 1,000 |
| 2023 | 3,789 | 1,000 |
| 2024 | 3,000 | 1,000 |
| 2025 | 3,000 | 1,000 |
| 2026 | 19,470 | – |

## How we got it

Two steps in Claude Code session `1e2ee420` (project `~/twicetold`)
made it.

1. On 2026-10-28, `uv run python -I src/areas.py` wrote the first five columns and `abstract_emb.npy`. The run that
   worked printed its result at 23:09 UTC (`44406 papers to embed`, `5-fold accuracy on 2026 labels: 0.593`).
   `src/areas.py` was added in commit `0b17fb8` and removed in commit `c57b0a8`. It:
   - takes the papers of `panels.jsonl` and the ICLR 2026 papers with a `primary_area` in `papers.parquet` (both
     the versions of that time);
   - makes a document of title, a newline and abstract, cut to 6,000 characters; drops documents of 50 characters
     or fewer;
   - embeds the documents with OpenAI `text-embedding-3-small` (a paid API) into `abstract_emb.npy`;
   - fits `LogisticRegression(max_iter=2000, C=3.0)` on the ICLR 2026 labels and predicts every row.
2. On 2026-10-28 at 23:30 UTC, an inline Python command (the same command that wrote `review_cdates.parquet` and
   `gptzero.parquet`, deleted on 2026-11-02) added three columns:
   - `area_true`: from `/tmp/scout/pa/dl/mlouis_<year>.parquet` (copy in `RAW_ML_REVIEWS_DIR/iclr_areas/`) for 2023–2026 (first `primary_area` per `forum`),
     joined on `paper_id`;
   - `area_final = area_true.fillna(area)`;
   - `area_source`.
3. Last write: 2026-10-28 23:30:35 UTC (file time).

The `panels.jsonl` and `papers.parquet` of 2026-10-28 no longer exist. On 2026-11-02 we copied
`/tmp/scout/pa/dl/` to `RAW_ML_REVIEWS_DIR/iclr_areas/` (byte-identical).

`src/areas.py classify` is the step-1 script without comments. `src/areas.py labels` is step 2.
A later version of step 1 (`src/corpus.py areas`) could not rebuild this file: it read `data/openreview_areas.parquet`
(never committed) and used different `area_true` and `area_final` rules. We removed it on 2026-11-02.

## Reproduce

```
uv run python -I src/areas.py classify --out <dir>
uv run python -I src/areas.py labels --areas <dir>/paper_areas.parquet --out <dir>/paper_areas_labelled.parquet
```

`classify` calls the paid OpenAI embedding API and reads `panels.jsonl` and `papers.parquet`. Those inputs changed
after 2026-10-28, so `classify` gives a different paper set now (44,399 papers, not 44,406). `labels` reads
`RAW_ML_REVIEWS_DIR/iclr_areas/` and makes no API call.

## Check

A full rebuild is not possible. On 2026-11-02 we ran these checks, with no API calls:

- `abstract_emb.npy` has 44,406 rows of 1,536 `float32` values; `row` runs from 0 to 44,405.
- All 44,406 paper ids are in the current `papers.parquet`. 30,936 of the 30,937 papers of `sample_reviews.parquet`
  have a row; the missing one is an ICLR 2017 paper.
- 19,470 rows are ICLR 2026 papers with a `primary_area` in the current `papers.parquet`. On all of them `area`
  equals that `primary_area`. All 6,439 rows where `area` differs from `area_pred` are among them.
- We refit the step-1 classifier on the saved `abstract_emb.npy` with the current ICLR 2026 labels. `area_pred`
  agrees on 44,328 of 44,406 rows (99.82%). `area_prob` differs by at most 0.0101. We do not know the cause of the
  78 differences. The labels of 2026-10-28 no longer exist, so we cannot compare them with the current labels.
- `src/areas.py labels` on the first five columns of this file gives a table identical to this file
  (`pandas.DataFrame.equals`).

## Read by

`src/panels.py` (`robust`: `area_pred` for `panels_placebo_area.jsonl`), `src/between.py` (`load_reviews`: `area_pred`).
