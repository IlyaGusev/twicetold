# data/nocq_reviews.parquet

ICLR 2023 sample reviews with the "clarity, quality, novelty and reproducibility" field removed. The pass-1 input for
`panels_nocq.jsonl`. One row per review: 14,271 rows, all reviews of the 3,789 ICLR 2023 panels of `panels.jsonl`.
Columns: `conf`, `year`, `paper_id`, `review_id`, `text`. `review_id` is `nocq_` + the OpenReview note id, so these
rows have their own ids in `claims_gpt-6-luna-high.jsonl`.

## Make

```
uv run python -I src/panels.py nocq
```

`scripts/inputs.sh` runs this step. The same step writes `panels_nocq.jsonl`. Deterministic, no API. It reads
`sample_reviews.parquet` (ICLR 2023 rows) and `panels.jsonl`. Every ICLR 2023 review text has two `[OTHER]` sections:
the review summary, then the clarity field (`corpus.SECTION_KEYS` order). The step cuts the text at the second
`[OTHER]` tag and stops with an error if a review does not have exactly two. The cut removes 16.6% of the characters.

## Notes

- Not in git (`.gitignore`). `scripts/inputs.sh` rebuilds it; `scripts/extract_all.sh` runs `inputs.sh` when the
  file is missing.

## Read by

`scripts/extract_all.sh`: pass-1 input, tag `_nocq`.
