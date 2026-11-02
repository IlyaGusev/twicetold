# data/other_pass1.parquet

The pass-1 input for the journal chain. One row per review: 40,992 reviews of 17,520 papers. Columns: `review_id`,
`text`.

| Source | Papers | Reviews |
|---|---|---|
| BMC-Med | 2,612 | 6,652 |
| Copernicus | 3,700 | 8,055 |
| F1000 | 4,400 | 10,496 |
| PLOS | 3,756 | 8,970 |
| SciPost | 3,052 | 6,819 |

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/panels.py other
```

It reads `other_reviews.parquet`, keeps at most 5 reviews per paper, drawn at random (`cap_reviews_per_paper`, seed
2027), and sorts by `conf`, `year`, `paper_id`, `review_id`. The same run writes `panels_other.jsonl`,
`panels_other_placebo.jsonl` and `reports/other_placebo_preprints.csv`. It is deterministic and needs no network.

## Notes

Git does not track this file. `scripts/extract_all.sh` runs `scripts/inputs.sh` when `data/noq_reviews.parquet` is
missing.

## Read by

`scripts/extract_all.sh` (pass-1 input, tag `_other`, `--paper_kind general`).
