# data/icml2025_reviews.parquet

ICML 2025 official reviews of accepted papers. One row per review: 12,478 rows of 3,257 papers.

| Column | Meaning |
|---|---|
| `conf`, `year` | `ICML`, 2025 |
| `paper_id`, `review_id` | OpenReview forum id and note id |
| `text` | Review form fields joined with tags (see below) |
| `rating` | `overall_recommendation` field |
| `confidence` | `confidence` field |
| `primary_area` | Primary area of the submission |
| `review_time`, `modified_time` | Note `cdate` and `mdate` in milliseconds |

Text tags: `[SUMMARY]`, `[CLAIMS AND EVIDENCE]`, `[METHODS AND EVALUATION]`, `[THEORY]`, `[EXPERIMENTS]`,
`[SUPPLEMENT]`, `[RELATED WORK]`, `[MISSING REFERENCES]`, `[STRENGTHS AND WEAKNESSES]`, `[OTHER]`, `[QUESTIONS]`. Empty
fields are left out.

## Make

```
uv run python -I src/fetch.py icml --year 2025
```

`scripts/inputs.sh` runs this step. It reads `icml2025_raw.jsonl` and keeps each reply whose invitation ends with
`/Official_Review`. Deterministic; no network when the raw file exists. No length filter and no deduplication;
`src/panels.py icml` does them.

## Notes

- Not in git (`.gitignore`). `scripts/inputs.sh` rebuilds it.

## Read by

`src/panels.py icml`.
