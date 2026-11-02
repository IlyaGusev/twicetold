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

## How we got it

The same run as `other_reviews.parquet` writes this file (`src/corpus.py other`, 2026-11-01 17:50 UTC). It merges
`data/other/<source>_papers.parquet` and keeps the papers that have 2 or more reviews.

## Reproduce

```
uv run python -I src/corpus.py other --out <dir> --report <dir>/other_inventory.txt
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw cache into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`).

## Read by

No code reads it. It documents titles and licences.
