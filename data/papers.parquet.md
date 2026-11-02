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

## Make

```
uv run python -I src/corpus.py ml
```

The same run writes `reviews.parquet`; see `reviews.parquet.md` for the raw inputs. No script in `scripts/` runs it.

- ICLR 2017–2025 and NeurIPS 2021–2025 come from `arcade/papers.parquet`.
- ICLR 2026 comes from `iclr2026/heineman.parquet` and `iclr2026/jerma_2026.parquet`. A paper whose final venue says
  `Withdrawn` or `Desk Rejected` gets that decision; otherwise the snapshot decision applies.

## Read by

`src/panels.py` (`ml`, `cocodet`), `src/outcomes.py` (`ml_panels`, `pairs`), `src/areas.py` (`classify`).
