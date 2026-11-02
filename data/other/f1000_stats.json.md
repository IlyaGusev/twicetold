# data/other/f1000_stats.json

Counts from the `src/fetch.py f1000` run that wrote `f1000_reviews.parquet`.

- `parsed_reviews`: 11,846
- `short_dropped`: 0
- `dup_dropped`: 0

`parsed_reviews` counts reviews before filters. `short_dropped` counts texts under 100 characters. `dup_dropped`
counts duplicate (paper, text) pairs.

## Make

`finalize` in `src/fetch.py` writes it in the same run as `f1000_reviews.parquet`. See `f1000_reviews.parquet.md`.

## Read by

`src/corpus.py other` (`inventory`) copies these counts into `reports/other_inventory.txt`.
