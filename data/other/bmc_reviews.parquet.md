# data/other/bmc_reviews.parquet

BMC Medicine first-round reviews. One row per review: 6,666 reviews of 2,612 papers. Columns: `conf`, `field`,
`journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`, `review_time`, `n_chars`.
`text` is the review body; `n_chars` is its length. `rating` and `rating_raw` are empty: BMC gives no rating.

| Year | Papers | Reviews |
|---|---|---|
| 2018 | 178 | 442 |
| 2019 | 193 | 454 |
| 2020 | 334 | 847 |
| 2021 | 327 | 893 |
| 2022 | 463 | 1,159 |
| 2023 | 485 | 1,265 |
| 2024 | 500 | 1,275 |
| 2025 | 132 | 331 |

## Make

```
uv run python -I src/fetch.py bmc
```

No run chain calls it. The same step writes `bmc_papers.parquet` and `bmc_stats.json`. No API cost.

- It reads `RAW_JOURNAL_REVIEWS_DIR/bmc/train.parquet`. When that file is missing, it downloads
  `https://huggingface.co/datasets/JerMa88/BMC_Medicine_Peer_Reviews/resolve/main/train.parquet` (branch `main`, not
  pinned).
- It keeps first-round reports only (`review_src` ends in `_V0_R<n>.pdf`).
- `text` is the part between `Reviewer's report:` and the first form question (for example "Are the methods
  appropriate").
- The date comes from the `Date:` line, the title from the `Title:` line.

`finalize` in `src/fetch.py` then applies the same rules to every source:

1. It drops reviews with fewer than 100 characters of text and duplicate (paper, text) pairs.
2. A paper's year is the year of its first review.
3. It keeps papers with 2 or more reviews and a year in 2018–2026.
4. It sorts the paper ids of each year, shuffles them with `random.Random(0)` and keeps the first 500.

## Notes

- Deterministic when the raw file exists. Without it, the step downloads again, and the source can differ.
- Git LFS stores it.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_reviews.parquet`.
