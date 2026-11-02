# data/other/scipost_reviews.parquet

SciPost (physics and related fields): one row per first-round review. 6,820 reviews of 3,052 papers.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`,
`review_time`, `n_chars`. `text` is the review body. `rating` is a source-specific integer (see below) and
`rating_raw` the source label. `n_chars` is the text length.

| Year | Papers | Reviews |
|---|---|---|
| 2018 | 110 | 246 |
| 2019 | 136 | 301 |
| 2020 | 229 | 518 |
| 2021 | 376 | 849 |
| 2022 | 433 | 955 |
| 2023 | 412 | 915 |
| 2024 | 449 | 996 |
| 2025 | 500 | 1,123 |
| 2026 | 407 | 917 |

## Make

```
uv run python -I src/fetch.py scipost
```

It writes `scipost_reviews.parquet`, `scipost_papers.parquet` and the `scipost` key of `stats.json` into `data/other/`
(`--out` changes the folder). No script in `scripts/` runs it. It needs network for pages that are not in the raw
cache `RAW_JOURNAL_REVIEWS_DIR/scipost/`; it never downloads a cached page again.

- It reads every page of the SciPost API (`https://scipost.org/api/submissions/`) and skips resubmissions.
- It keeps reports with status `vetted` and type `report_normal`.
- `text` joins the fields strengths, weaknesses, report and requested changes with the tags `[STRENGTHS]`,
  `[WEAKNESSES]`, `[REPORT]`, `[REQUESTED CHANGES]`.
- `rating` is the rounded mean of validity, significance, originality and clarity on a 1–6 scale
  (`poor` = 1 … `top` = 6). `rating_raw` lists all six criteria.
- `year` starts as the submission year. Every paper has licence CC BY 4.0.

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
