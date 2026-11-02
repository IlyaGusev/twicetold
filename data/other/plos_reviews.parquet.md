# data/other/plos_reviews.parquet

PLOS journals (ONE, Biology, Computational Biology, Genetics, Medicine, Pathogens, NTD, Global Public Health,
Digital Health): one row per first-round review. 9,015 reviews of 3,756 papers.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`,
`review_time`, `n_chars`. `text` is the review body. `rating` is a source-specific integer (see below) and
`rating_raw` the source label. `n_chars` is the text length.

| Year | Papers | Reviews |
|---|---|---|
| 2019 | 500 | 1,174 |
| 2020 | 500 | 1,206 |
| 2021 | 500 | 1,181 |
| 2022 | 500 | 1,176 |
| 2023 | 500 | 1,208 |
| 2024 | 500 | 1,191 |
| 2025 | 500 | 1,242 |
| 2026 | 256 | 637 |

## Make

```
uv run python -I src/fetch.py plos
```

It writes `plos_reviews.parquet`, `plos_papers.parquet` and `plos_stats.json` into `data/other/` (`--out`
changes the folder). No script in `scripts/` runs it. It needs network for items that are not in the raw cache
`RAW_JOURNAL_REVIEWS_DIR/plos/`; it never downloads a cached item again.

- For each year 2018–2026 it searches Europe PMC for the nine journals with the phrase `"Decision Letter 0"`, then
  downloads the full-text XML.
- It skips papers that are resubmissions and papers whose earlier reviews appear in an author response.
- It takes the first decision letter that contains `Reviewer #<n>:` and splits it into one text per reviewer. It
  removes the editor's text and the journal boilerplate.
- Short yes/no answers to the checklist go to `rating_raw`. `rating` uses the "technically sound" question:
  Yes = 3, Partly = 2, No = 1.
- `field` is `multidisciplinary` for PLOS ONE and `biomed` for the others.

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
