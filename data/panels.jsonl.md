# data/panels.jsonl

The main panels: one panel per sampled ICLR or NeurIPS paper. 30,930 panels with 116,419 reviews. Keys:
`panel_id` (the OpenReview forum id, no prefix) and `review_ids`.

Panel sizes: 3 reviews (10,049), 4 reviews (18,133), 5 reviews (2,748).

## How we got it

1. The same run as `sample_reviews.parquet` writes this file: `uv run python -I src/panels.py ml --data
   data/backup_predup --current_corpus data` (`src/panels.py`, function `ml`).
2. It keeps the reviews of `sample_reviews.parquet` that are also in the current `data/reviews.parquet`. It groups
   them by paper (`by_paper_panels`) and keeps papers with 3 or more such reviews.
3. Last write: 2026-11-01 17:27:52 UTC, step `src/panels.py ml` of `regen.sh` in Claude Code session `881519d6`
   (log `logs/regen_gpt-6-luna-high.log`).

The sample came from the earlier build of `reviews.parquet` in `data/backup_predup/`. That folder was deleted on
2026-11-02 and no copy exists. `panels.py ml` cannot draw the sample again. The step from the sample to the panels
needs only `sample_reviews.parquet` and the current `reviews.parquet`, so we can still check it.

## Reproduce

The full command cannot run now (see `sample_reviews.parquet.md`). The panel step alone is the last part of
`panels.py ml`: `by_paper_panels` on the rows of `sample_reviews.parquet` whose `review_id` is in
`reviews.parquet`, then keep panels with 3 or more reviews.

## Check

On 2026-11-02 we ran the panel step of `ml` (the code above, from `src/panels.py`) on `sample_reviews.parquet` and
the current `reviews.parquet` into a scratch folder. The output is byte-identical to this file.

Also: all 116,419 review ids are in `sample_reviews.parquet` and in the current `reviews.parquet`. 14 sample reviews
that are still in `reviews.parquet` are not in a panel: their papers have fewer than 3 such reviews.

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain), `src/panels.py` (`robust`, `sol`), `src/outcomes.py`
(`pair_design`, through `PANEL_FILES`), `src/corpus.py` (`areas`).
