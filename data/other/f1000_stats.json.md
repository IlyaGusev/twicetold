# data/other/f1000_stats.json

Counts from the fetch run of f1000. The same run as `f1000_reviews.parquet` writes it (`finalize` in
`src/fetch.py`).

- `parsed_reviews`: 11,846
- `short_dropped`: 0
- `dup_dropped`: 0

`parsed_reviews` counts reviews before filters; `short_dropped` counts texts under 100 characters; `dup_dropped`
counts duplicate (paper, text) pairs. Other keys count papers that the parser skipped, by reason.

## Read by

`src/corpus.py other` copies these counts into `reports/other_inventory.txt`.
