# data/cocodet_input_other.parquet

The journal reviews that CoCoDet scores. One row per review: 10,042 rows, no repeated id. Columns: `review_id`,
`text` (strings).

The rows are all reviews of up to 100 random same-paper (non-placebo) panels per journal source and year. Reviews per
source, by a join with the current `other_reviews.parquet`: F1000 2,166; BMC-Med 2,017; SciPost 1,970; Copernicus
1,849; PLOS 1,745; not found 295.

## Make

No command in the repo makes this file. It is a saved sample, and the repo does not hold its source panel table.

## Notes

- 295 ids are not in the current `other_reviews.parquet`.
- Git LFS stores it.

## Read by

`src/cocodet.py scores --input data/cocodet_input_other.parquet`, which writes `cocodet_other.jsonl`. No analysis
module reads it.
