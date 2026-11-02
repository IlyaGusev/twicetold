# data/icml2026_reviews.parquet

ICML 2026 official reviews of accepted papers. One row per review: 24,378 rows of 6,341 papers.

| Column | Meaning |
|---|---|
| `conf`, `year` | `ICML`, 2026 |
| `paper_id`, `review_id` | OpenReview forum id and note id |
| `text` | Review form fields joined with tags (see below) |
| `rating` | `overall_recommendation` field |
| `confidence` | `confidence` field |
| `primary_area` | Primary area of the submission |
| `review_time`, `modified_time` | Note `cdate` and `mdate` in milliseconds |

Text tags: `[SUMMARY]`, `[STRENGTHS AND WEAKNESSES]`, `[QUESTIONS]`, `[LIMITATIONS]`. Empty fields are left out.

## Make

```
uv run python -I src/fetch.py icml --year 2026
```

`scripts/inputs.sh` runs this step. It reads `icml2026_raw.jsonl` and keeps each reply whose invitation ends with
`/Official_Review`. Deterministic; no network when the raw file exists. No length filter and no deduplication;
`src/panels.py icml` does them.

## Notes

- Not in git (`.gitignore`). `scripts/inputs.sh` rebuilds it.

## Read by

`src/panels.py icml`.
