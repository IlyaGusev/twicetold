# data/panels.jsonl

The main panels: one panel per sampled ICLR or NeurIPS paper. 30,930 panels with 116,419 reviews. Keys: `panel_id`
(the OpenReview forum id, no prefix) and `review_ids`.

Panel sizes: 3 reviews (10,049), 4 reviews (18,133), 5 reviews (2,748).

Each panel holds the reviews of one paper of `sample_reviews.parquet` that are also in `reviews.parquet`. Papers
with fewer than 3 such reviews have no panel.

## Make

No command in the repo rebuilds this file. `uv run python -I src/panels.py ml` writes it together with
`sample_reviews.parquet` and `panels_placebo.jsonl`, but on the current `reviews.parquet` it draws a different
sample.

## Notes

14 sample reviews that are in `reviews.parquet` are not in a panel: their papers have fewer than 3 such reviews.

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain), `src/panels.py` (`robust`, `cocodet`), `src/areas.py`
(`classify`).
