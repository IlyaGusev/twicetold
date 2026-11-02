# data/noq_reviews.parquet

ICLR 2024–2026 sample reviews with the questions field removed. The pass-1 input for `panels_noq.jsonl`. One row per
review: 46,972 rows. Columns: `conf`, `year`, `paper_id`, `review_id`, `text`.

| Year | Papers | Reviews |
|---|---|---|
| 2024 | 3,000 | 11,596 |
| 2025 | 3,000 | 12,046 |
| 2026 | 6,000 | 23,330 |

## Make

```
uv run python -I src/panels.py robust
```

`scripts/inputs.sh` runs this step. The same step writes `panels_placebo_area.jsonl`, `panels_sub3.jsonl` and
`panels_noq.jsonl`. Deterministic (seeds 1 and 11), no API. It reads `sample_reviews.parquet` (ICLR rows only),
`paper_areas.parquet` and `panels.jsonl`. The rows are all ICLR 2024–2026 reviews of `sample_reviews.parquet`: first
the reviews of 400 random papers per year, then the rest. `drop_questions` removes the text from `[QUESTIONS]` up to
the next `[LIMITATIONS]` or `[OTHER]` tag, or to the end.

## Notes

- Not in git (`.gitignore`). `scripts/inputs.sh` rebuilds it; `scripts/extract_all.sh` runs `inputs.sh` when the
  file is missing.

## Read by

`scripts/extract_all.sh`: pass-1 input, tag `_noq`.
