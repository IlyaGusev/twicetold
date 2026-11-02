# data/other/bmc_papers.parquet

BMC Medicine: one row per paper in `bmc_reviews.parquet`. 2,612 papers. Columns: `conf`, `field`, `journal`,
`paper_id`, `title`, `licence`, `year`, `n_reviews`. `n_reviews` counts the paper's rows in `bmc_reviews.parquet`;
`licence` is the article licence (CC BY 4.0).

## Make

```
uv run python -I src/fetch.py bmc
```

The same step writes `bmc_reviews.parquet` and `bmc_stats.json`; see `bmc_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_papers.parquet`.
