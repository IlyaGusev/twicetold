# data/other_papers.parquet

Papers behind `other_reviews.parquet`. One row per paper: 17,520 rows.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `title`, `n_reviews`, `licence`. `n_reviews` counts the
paper's rows in `other_reviews.parquet`.

| Year | BMC-Med | Copernicus | F1000 | PLOS | SciPost |
|---|---|---|---|---|---|
| 2018 | 178 | 300 | 450 | 0 | 110 |
| 2019 | 193 | 300 | 500 | 500 | 136 |
| 2020 | 334 | 300 | 500 | 500 | 229 |
| 2021 | 327 | 300 | 500 | 500 | 376 |
| 2022 | 463 | 500 | 450 | 500 | 433 |
| 2023 | 485 | 500 | 500 | 500 | 412 |
| 2024 | 500 | 500 | 500 | 500 | 449 |
| 2025 | 132 | 500 | 500 | 500 | 500 |
| 2026 | 0 | 500 | 500 | 256 | 407 |

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/corpus.py other
```

It merges `data/other/<source>_papers.parquet` and keeps the papers that have 2 or more reviews in
`other_reviews.parquet`. The same run writes `other_reviews.parquet`. It is deterministic and needs no network.

## Notes

Git does not track this file. It documents titles and licences.

## Read by

Nothing reads it.
