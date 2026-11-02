# data/review_cdates.parquet

Creation time and rating before and after the rebuttal for ICLR 2023–2025 reviews. 71,415 rows.

| Column | Meaning |
|---|---|
| `review_id` | Key into `reviews.parquet` |
| `review_cdate` | Review creation time |
| `ext_rating_before`, `ext_rating_after` | Rating before and after the rebuttal; only 2024 and 2025 have them |

Rows by year (through `reviews.parquet`): 2023: 14,327; 2024: 22,098; 2025: 34,990. All rows match a review.

## How we got it

No script in the repo makes this file.

1. On 2026-10-28 at 22:53 UTC, a subagent in Claude Code session `1e2ee420` (project `~/twicetold`) downloaded
   `iclr_{2023,2024,2025,2026}_reviews.parquet` from Hugging Face `MlouisBE/iclr-rebuttal-analysis`
   (`https://huggingface.co/datasets/MlouisBE/iclr-rebuttal-analysis/resolve/main/data/processed/parquet/iclr_<year>_reviews.parquet`)
   into `/tmp/scout/pa/dl/mlouis_<year>.parquet`.
2. On 2026-10-28 at 23:30 UTC, an inline Python command in the same session made the file:
   - reviewer label = `"Reviewer_" + reviewer_suffix`;
   - inner join with the ICLR rows of `reviews.parquet` (the version of that time) on `paper_id`, `reviewer` and
     `year`;
   - `review_cdate` converted from milliseconds;
   - columns `review_id`, `review_cdate`, `ext_rating_before`, `ext_rating_after` saved.

   The same command also wrote `gptzero.parquet` and changed `paper_areas.parquet`.

The files in `/tmp/scout/pa/dl/` still exist (2026-11-02), but `/tmp` is not permanent.

## Read by

`src/outcomes.py` (`review_meta`): a review counts as edited when `ext_rating_before != ext_rating_after`.
