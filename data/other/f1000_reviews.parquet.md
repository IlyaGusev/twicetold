# data/other/f1000_reviews.parquet

F1000Research, Wellcome Open Research and Open Research Europe: one row per first-round review. 10,505 reviews of
4,400 papers.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`,
`review_time`, `n_chars`. `text` is the review body. `rating` is a source-specific integer (see below) and
`rating_raw` the source label. `n_chars` is the text length.

| Year | Papers | Reviews |
|---|---|---|
| 2018 | 450 | 1,094 |
| 2019 | 500 | 1,174 |
| 2020 | 500 | 1,159 |
| 2021 | 500 | 1,130 |
| 2022 | 450 | 1,003 |
| 2023 | 500 | 1,202 |
| 2024 | 500 | 1,235 |
| 2025 | 500 | 1,264 |
| 2026 | 500 | 1,244 |

## Make

```
uv run python -I src/fetch.py f1000
```

It writes `f1000_reviews.parquet`, `f1000_papers.parquet` and `f1000_stats.json` into `data/other/` (`--out`
changes the folder). No script in `scripts/` runs it. It needs network for items that are not in the raw cache
`RAW_JOURNAL_REVIEWS_DIR/f1000/`; it never downloads a cached item again.

- For each year 2018–2026 it searches Europe PMC for the three journals and downloads the full-text XML of each hit.
- It takes reviewer reports on version 1 of each article only.
- `text` is the report body without references, yes/no checklist answers, the reviewer expertise line and the
  confirmation statement.
- `rating`: Approved = 3, Approved with Reservations = 2, Not Approved = 1. `rating_raw` keeps the label.
- It scans candidates in shuffled order (seed 0), 100 at a time, until it finds 500 papers with 2 or more reviews
  for the year.

`finalize` in `src/fetch.py` then applies the same rules to every source:

1. It drops texts under 100 characters and duplicate (paper, text) pairs.
2. It sets a paper's year to the year of its first review.
3. It keeps papers with 2 or more reviews and a year in 2018–2026.
4. Per year, it sorts the paper ids, shuffles them with `random.Random(0)` and keeps the first 500.

## Notes

With a full raw cache the output is deterministic. Without the cache the command downloads again; the sources can
change, so the rows can differ.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_reviews.parquet`.
