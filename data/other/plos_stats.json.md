# data/other/plos_stats.json

Counts from the `src/fetch.py plos` run that wrote `plos_reviews.parquet`.

- `resubmission`: 138
- `earlier_reviews`: 3
- `letters`: 5,086
- `letters_split_2plus`: 4,500
- `parsed_reviews`: 11,251
- `short_dropped`: 0
- `dup_dropped`: 1

`resubmission` and `earlier_reviews` count papers that the parser skipped, by reason. `letters` counts parsed
decision letters; `letters_split_2plus` counts the letters that split into 2 or more reviewer texts.
`parsed_reviews` counts reviews before filters. `short_dropped` counts texts under 100 characters. `dup_dropped`
counts duplicate (paper, text) pairs.

## Make

`finalize` in `src/fetch.py` writes it in the same run as `plos_reviews.parquet`. See `plos_reviews.parquet.md`.

## Read by

`src/corpus.py other` (`inventory`) copies these counts into `reports/other_inventory.txt` and reports
`letters_split_2plus` / `letters`.
