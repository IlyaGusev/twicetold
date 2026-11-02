# data/paper_areas.parquet

Primary area for each sampled ICLR and NeurIPS paper and each ICLR 2026 paper with an author-chosen area. One row per
paper: 44,406 rows.

| Column | Meaning |
|---|---|
| `paper_id` | OpenReview forum id |
| `area` | Author-chosen ICLR 2026 primary area when the paper has one; otherwise `area_pred` |
| `area_prob` | Classifier probability of `area_pred` |
| `area_pred` | Area the classifier predicts from title and abstract |
| `row` | Row of the paper in the embedding matrix of `classify`, which the command does not save (equal to the row number here) |
| `area_true` | Primary area from the `MlouisBE/iclr-rebuttal-analysis` files (ICLR 2023–2026); empty otherwise |
| `area_final` | `area_true` when present; otherwise `area` |
| `area_source` | `openreview` when `area_true` is present (29,249 rows); otherwise `predicted` (15,157 rows) |

There are 21 areas. Rows per venue and year (venue and year from `papers.parquet`):

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

## Make

No command in the repo rebuilds this exact file. Two steps in `src/areas.py` make a file of this form:

```
uv run python -I src/areas.py classify --out <dir>
uv run python -I src/areas.py labels --areas <dir>/paper_areas.parquet --out <dir>/paper_areas_labelled.parquet
```

- `classify` reads `panels.jsonl` and `papers.parquet`. It takes the panel papers and the ICLR 2026 papers with a
  `primary_area`, and makes a document of title, a newline and abstract, cut to 6,000 characters. It drops documents
  of 50 characters or fewer. It embeds the documents with OpenAI `text-embedding-3-small` (paid API, needs
  `OPENAI_API_KEY`); it does not save the vectors. It fits `LogisticRegression(max_iter=2000, C=3.0)` on the ICLR 2026
  labels and predicts every row. It writes the first five columns.
- `labels` adds `area_true`, `area_final` and `area_source`. It reads `ICLR_AREAS_DIR/mlouis_<year>.parquet` (first
  `primary_area` per `forum`) and joins on `paper_id`. It is deterministic and makes no API call.

## Notes

On the current `panels.jsonl` and `papers.parquet`, `classify` selects 44,399 papers, not 44,406, so a rerun gives a
different file. `labels` on the first five columns of this file gives this file.

## Read by

`src/panels.py` (`robust`: `area_pred` for `panels_placebo_area.jsonl`), `src/between.py` (`load_reviews`:
`area_pred`).
