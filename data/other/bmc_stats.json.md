# data/other/bmc_stats.json

Counts from the fetch run of bmc. The same run as `bmc_reviews.parquet` writes it (`finalize` in
`src/fetch.py`).

- `parsed_reviews`: 8,109
- `short_dropped`: 302
- `dup_dropped`: 2

`parsed_reviews` counts reviews before filters; `short_dropped` counts texts under 100 characters; `dup_dropped`
counts duplicate (paper, text) pairs. Other keys count papers that the parser skipped, by reason.

## Read by

`src/corpus.py other` copies these counts into `reports/other_inventory.txt`.
