# data/panels_placebo.jsonl

Cross-paper control panels for the main sample. Each panel holds 4 reviews of 4 different papers of one venue and
year. 4,500 panels: 300 for each of the 15 venue-year cells. Keys: `panel_id` (`placebo_<conf>_<year>_<k>`) and
`review_ids`. The panels hold 18,000 review ids, 16,042 of them unique.

All review ids are in `sample_reviews.parquet`. 17,812 of the 18,000 are in `reviews.parquet`.

## Make

No command in the repo rebuilds this file. `uv run python -I src/panels.py ml` writes it together with
`sample_reviews.parquet` and `panels.jsonl`, but on the current `reviews.parquet` it draws a different sample.

The panel rule (`ml`, inner function `placebo`): for each (conf, year) cell of the sample, 300 times, pick 4 papers
at random, then 1 review of each paper at random.

## Notes

`src/outcomes.py` drops review ids that are not in `reviews.parquet` from a panel and keeps the panel if 2 or more
reviews stay.

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain), `src/gold.py` (`sample`: the cross-paper panels of the
reference set).
