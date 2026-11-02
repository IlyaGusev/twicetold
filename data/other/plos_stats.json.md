# data/other/plos_stats.json

Counts from the fetch run of plos. The same run as `plos_reviews.parquet` writes it (`finalize` in
`src/fetch.py`).

- `resubmission`: 138
- `earlier_reviews`: 3
- `letters`: 5,086
- `letters_split_2plus`: 4,500
- `parsed_reviews`: 11,251
- `short_dropped`: 0
- `dup_dropped`: 1

`parsed_reviews` counts reviews before filters; `short_dropped` counts texts under 100 characters; `dup_dropped`
counts duplicate (paper, text) pairs. Other keys count papers that the parser skipped, by reason.

## Read by

`src/corpus.py other` copies these counts into `reports/other_inventory.txt`.
