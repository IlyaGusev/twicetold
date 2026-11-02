# data/cocodet_input_other.parquet

The journal reviews that CoCoDet scored. One row per review: 10,042 rows. Columns: `review_id`, `text` (strings).
Rows are in the order of `other_reviews.parquet` of 2026-10-29.

The rows are all reviews of up to 100 random non-placebo panels per source and year. Reviews per source, by a join
with the current `other_reviews.parquet`: F1000 2,166; BMC-Med 2,017; SciPost 1,970; Copernicus 1,849; PLOS 1,745;
not found 295.

## How we got it

No script in the repo makes this file.

On 2026-10-29 at 21:59:26 UTC, an inline Python command in Claude Code session `881519d6` (project `~/ideas`) wrote
it:

1. It read `data/other_panel_outcomes_gpt-6-luna.parquet` and kept rows with `placebo == False`.
2. For each (`conf`, `year`) it drew `min(100, n)` panels with `sample(random_state=0)`.
3. It took the review ids of those panels from `data/panels_other.jsonl`.
4. It kept the rows of `data/other_reviews.parquet` (`review_id`, `text`) with those ids.

`other_panel_outcomes_gpt-6-luna.parquet` no longer exists. `other_reviews.parquet` and `panels_other.jsonl` were
rebuilt on 2026-11-01.

## Reproduce

We cannot reproduce this file: the panel table of step 1 is gone.

## Check

On 2026-11-02 we compared this file with the current tables:

- 10,042 rows, no duplicate ids.
- 9,747 ids are in the current `other_reviews.parquet`; 295 are not. 9,714 of the 9,747 texts are equal.
- 9,741 ids are in a panel of the current `panels_other.jsonl`. They cover 4,172 panels; 4,167 of these are
  complete. No source and year has more than 100 panels.

## Read by

`src/ai.py cocodet` (input). No analysis module reads it.
