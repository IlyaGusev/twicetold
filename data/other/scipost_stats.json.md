# data/other/scipost_stats.json

Counts from the `src/fetch.py scipost` run that wrote `scipost_reviews.parquet`.

- `api_missing`: 3
- `parsed_reviews`: 8,603
- `short_dropped`: 209
- `dup_dropped`: 15

`api_missing` counts submissions that the API pages did not return. `parsed_reviews` counts reviews before filters.
`short_dropped` counts texts under 100 characters. `dup_dropped` counts duplicate (paper, text) pairs.

## Make

`finalize` in `src/fetch.py` writes it in the same run as `scipost_reviews.parquet`. See
`scipost_reviews.parquet.md`.

## Read by

`src/corpus.py other` (`inventory`) copies these counts into `reports/other_inventory.txt`.
