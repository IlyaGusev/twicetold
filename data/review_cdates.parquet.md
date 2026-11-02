# data/review_cdates.parquet

Creation time and rating before and after the rebuttal for ICLR 2023–2025 reviews. 71,415 rows.

| Column | Meaning |
|---|---|
| `review_id` | Key into `reviews.parquet` |
| `review_cdate` | Review creation time |
| `ext_rating_before`, `ext_rating_after` | Rating before and after the rebuttal; only 2024 and 2025 have them |

Rows by year (through `reviews.parquet`): 2023: 14,327; 2024: 22,098; 2025: 34,990. All rows match a review.

## How we got it

1. On 2026-10-28 at 22:53 UTC, a subagent in Claude Code session `1e2ee420` (project `~/twicetold`) downloaded
   `iclr_{2023,2024,2025,2026}_reviews.parquet` from Hugging Face `MlouisBE/iclr-rebuttal-analysis`
   (`https://huggingface.co/datasets/MlouisBE/iclr-rebuttal-analysis/resolve/main/data/processed/parquet/iclr_<year>_reviews.parquet`)
   into `/tmp/scout/pa/dl/mlouis_<year>.parquet`. A copy is in `RAW_ML_REVIEWS_DIR/iclr_areas/`.
2. On 2026-10-28 at 23:30 UTC, an inline Python command in the same session made the file:
   - reviewer label = `"Reviewer_" + reviewer_suffix`;
   - inner join with the ICLR rows of `reviews.parquet` (the version of that time) on `paper_id`, `reviewer` and
     `year`;
   - `review_cdate` converted from milliseconds;
   - columns `review_id`, `review_cdate`, `ext_rating_before`, `ext_rating_after` saved.

   The same command also wrote `gptzero.parquet` (deleted on 2026-11-02) and changed `paper_areas.parquet`.

On 2026-11-02 we copied `/tmp/scout/pa/dl/` to `RAW_ML_REVIEWS_DIR/iclr_areas/` (byte-identical).

## Reproduce

```
uv run python -I src/external.py review_cdates --out <dir>/review_cdates.parquet
```

`src/external.py review_cdates` is the step-2 code without the GPTZero and area parts.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current `reviews.parquet` into a scratch folder. The output is
identical to this file (`pandas.DataFrame.equals`).

## Read by

`src/outcomes.py` (`review_meta`): a review counts as edited when `ext_rating_before != ext_rating_after`.
