# data/other/copernicus_stats.json

Counts from the fetch run of copernicus. The same run as `copernicus_reviews.parquet` writes it (`finalize` in
`src/fetch.py`).

- `pdf with manuscript pages`: 92
- `parsed_reviews`: 9,682
- `short_dropped`: 34
- `dup_dropped`: 0

`parsed_reviews` counts reviews before filters; `short_dropped` counts texts under 100 characters; `dup_dropped`
counts duplicate (paper, text) pairs. Other keys count papers that the parser skipped, by reason.

## Read by

`src/corpus.py other` copies these counts into `reports/other_inventory.txt`.
