# data/icml2026_reviews.parquet

ICML 2026 official reviews of accepted papers. One row per review: 24,378 rows.

| Column | Meaning |
|---|---|
| `conf`, `year` | `ICML`, 2026 |
| `paper_id`, `review_id` | OpenReview forum id and note id |
| `text` | Review form fields joined with tags (see below) |
| `rating` | `overall_recommendation` field |
| `confidence` | `confidence` field |
| `primary_area` | Primary area of the submission |
| `review_time`, `modified_time` | Note `cdate` and `mdate` in milliseconds |

Text tags for 2026: `[SUMMARY]`, `[STRENGTHS AND WEAKNESSES]`, `[QUESTIONS]`, `[LIMITATIONS]`. Empty fields are left out.

## How we got it

`uv run python -I src/fetch.py icml --year 2026` (`icml_reviews` in `src/fetch.py`). It reads
`icml2026_raw.jsonl` and keeps each reply whose invitation ends with `/Official_Review`. No length filter
and no deduplication happen here; `src/panels.py icml` does them later.

Last write: 2026-10-29 at about 14:41 UTC, session `881519d6`.

## Reproduce

```
uv run python -I src/fetch.py icml --year 2026 --out <dir>
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw cache into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`).

## Read by

`src/panels.py icml`, which writes `icml_sample.parquet`, `icml_pass1.parquet` and `panels_icml.jsonl`.
