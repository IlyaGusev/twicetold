# data/gold/tune_reviews.parquet

The 240 review texts of the reference set (60 panels × 4). One row per review. Columns: `review_id`, `text`,
`conf`, `year`, `paper_id`.

## Make

```
uv run python -I src/gold.py sample --out <dir>
```

Step 1 of `scripts/gold.sh <dir>`. The same step writes `panels_tune.jsonl`. Deterministic, no API. It takes the
rows of `data/sample_reviews.parquet` that are in a same-paper panel (columns `review_id`, `text` only), then all of
`sample_reviews.parquet`, keeps the 240 panel reviews and drops duplicate ids. `tests/test_gold.py` rebuilds the file
and checks that it is equal.

## Notes

- The 160 reviews of same-paper panels have empty `conf`, `year` and `paper_id`, because the step keeps their
  two-column rows.

## Read by

`src/gold.py council_claims` (as the `src/extract.py pass1` input), `src/gold.py claim_inputs`.
