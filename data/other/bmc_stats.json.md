# data/other/bmc_stats.json

Filter counts of the BMC fetch:

- `parsed_reviews`: 8,109 reviews before filters;
- `short_dropped`: 302 texts with fewer than 100 characters;
- `dup_dropped`: 2 duplicate (paper, text) pairs.

## Make

```
uv run python -I src/fetch.py bmc
```

`finalize` in `src/fetch.py` writes it in the same step as `bmc_reviews.parquet`; see `bmc_reviews.parquet.md`.

## Read by

`src/corpus.py other` copies these counts into `reports/other_inventory.txt`.
