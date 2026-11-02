# data/cocodet_input.parquet

The ICLR and NeurIPS reviews that CoCoDet scores. One row per review: 26,484 reviews of 7,000 papers. Columns:
`review_id`, `text` (both strings, copied from `sample_reviews.parquet`). Rows are in `sample_reviews.parquet` order.

The rows are all reviews of a random subset of the panels in `panels.jsonl`:

| Year | ICLR papers | ICLR reviews | NeurIPS papers | NeurIPS reviews |
|---|---|---|---|---|
| 2017 | 300 | 946 | – | – |
| 2018 | 300 | 905 | – | – |
| 2019 | 300 | 916 | – | – |
| 2020 | 300 | 921 | – | – |
| 2021 | 300 | 1,154 | 300 | 1,167 |
| 2022 | 300 | 1,173 | 300 | 1,101 |
| 2023 | 300 | 1,113 | 300 | 1,315 |
| 2024 | 1,000 | 3,862 | 500 | 1,958 |
| 2025 | 1,000 | 4,025 | 500 | 2,026 |
| 2026 | 1,000 | 3,902 | – | – |

## Make

`scripts/inputs.sh` runs this step:

```
uv run python -I src/panels.py cocodet
```

It groups the panels of `panels.jsonl` by venue and year (from `papers.parquet`), draws panels without replacement
with `np.random.default_rng(7)` (counts in `panels.COCODET_PANELS`), and keeps the rows of `sample_reviews.parquet`
whose review is in a drawn panel. Deterministic, no network, no paid API.

## Notes

- Not in git (`.gitignore`). `scripts/inputs.sh` rebuilds it.

## Read by

`src/cocodet.py scores --input data/cocodet_input.parquet`, which writes `cocodet_main.jsonl`. No analysis module reads
it.
