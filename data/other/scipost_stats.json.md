# data/other/scipost_stats.json

Counts from the fetch run of scipost. The same run as `scipost_reviews.parquet` writes it (`finalize` in
`src/fetch.py`).

- `api_missing`: 3
- `parsed_reviews`: 8,603
- `short_dropped`: 209
- `dup_dropped`: 15

`parsed_reviews` counts reviews before filters; `short_dropped` counts texts under 100 characters; `dup_dropped`
counts duplicate (paper, text) pairs. Other keys count papers that the parser skipped, by reason.

## Read by

`src/corpus.py other` copies these counts into `reports/other_inventory.txt`.
