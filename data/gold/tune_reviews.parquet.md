# data/gold/tune_reviews.parquet

The 240 review texts of the reference set (60 panels × 4). Columns: `review_id`, `text`, `conf`, `year`, `paper_id`.

The 160 reviews of same-paper panels have empty `conf`, `year` and `paper_id`. The original command took them from
`data/blind_reviews.parquet`, which held only `review_id` and `text`, and put that file first.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

`src/gold.py sample`: rows of `data/sample_reviews.parquet` whose review is in a blind panel (columns `review_id`,
`text`), followed by all of `sample_reviews.parquet`; it keeps the 240 panel reviews and drops duplicate ids.
`data/blind_reviews.parquet` no longer exists; the code makes the same rows from `sample_reviews.parquet`.

Original run: 2026-10-31 14:46 UTC.

## Check

`tests/test_gold.py` rebuilds this file from the inputs above. The result is byte-identical.

## Read by

`src/extract.py pass1` (council claims), `src/gold.py claim_inputs`.
