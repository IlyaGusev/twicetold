# data/wd_reviews.parquet

Reviews of withdrawn ICLR 2026 papers and the pass-1 input of the withdrawn-paper check. One row per review: 3,891
reviews of 1,000 papers. Columns: `conf` (`ICLR`), `year` (2026), `paper_id`, `review_id`, `text`.

Reviews per paper: 3 (164 papers), 4 (781), 5 (55).

The papers are 1,000 random ICLR 2026 papers with `withdrawn` true and 3 or more reviews
(`numpy.random.default_rng(7)`). Each paper keeps at most 5 reviews, in random order.

## Make

No command in the repo makes this file. It is a saved one-off output. A new draw on the current `reviews.parquet`
picks other papers.

## Notes

All 3,891 reviews are in `reviews.parquet`, with the same text.

## Read by

`scripts/extract_all.sh` (pass-1 input, tag `_wd`), `src/panels.py wd`.
