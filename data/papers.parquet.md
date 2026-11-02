# data/papers.parquet

ICLR 2017–2026 and NeurIPS 2021–2025 papers. One row per OpenReview forum: 66,734 rows.

| Column | Meaning |
|---|---|
| `conf`, `year`, `paper_id` | Venue, year, OpenReview forum id |
| `title`, `abstract` | From the source dump |
| `decision_raw` | Decision text as the source gives it |
| `decision` | `accept`, `reject`, `withdrawn` or `desk_reject` (`corpus.py:decision_class`) |
| `withdrawn` | `decision == "withdrawn"` |
| `primary_area` | Author-chosen area; only ICLR 2026 has it |
| `final_venue` | Final venue string; only ICLR 2026 has it |
| `n_reviews` | Reviews of this paper in `reviews.parquet` |

Rows per venue and year:

| Year | ICLR | NeurIPS |
|---|---|---|
| 2017 | 490 | – |
| 2018 | 935 | – |
| 2019 | 1,419 | – |
| 2020 | 2,213 | – |
| 2021 | 2,594 | 2,768 |
| 2022 | 2,617 | 2,824 |
| 2023 | 3,793 | 3,394 |
| 2024 | 5,750 | 4,236 |
| 2025 | 8,701 | 5,529 |
| 2026 | 19,471 | – |

## How we got it

The same run as `reviews.parquet` writes this table; see `reviews.parquet.md` for the downloads.

- ICLR 2017–2025 and NeurIPS 2021–2025 come from `arcade/papers.parquet` (Hugging Face
  `ulab-ai/ResearchArcade-openreview-papers`).
- ICLR 2026 comes from `iclr2026/heineman.parquet` and `iclr2026/jerma_2026.parquet`. A paper whose final venue
  says `Withdrawn` or `Desk Rejected` gets that decision; otherwise the snapshot decision applies.

Last write: 2026-11-01 17:27 UTC, step `src/corpus.py ml` of `regen.sh` (session `881519d6`).

## Reproduce

```
uv run python -I src/corpus.py ml --out <dir>
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw dumps in `RAW_ML_REVIEWS_DIR` into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`).

## Read by

`src/panels.py` (`ml`), `src/outcomes.py`, `src/areas.py` (`classify`).
