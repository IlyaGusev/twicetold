# data/panels_wd.jsonl

Panels of withdrawn ICLR 2026 papers: one panel per paper of `wd_reviews.parquet`. 1,000 panels with 3,891 reviews.
Keys: `panel_id` (`wd_<paper_id>`) and `review_ids`. Sorted by paper id.

Panel sizes: 3 reviews (164), 4 (781), 5 (55).

Each panel holds the reviews of one paper of `wd_reviews.parquet`, without review `zlaasankug` (same text as review
`foBgDN8Ecm` of the same paper). All 3,891 review ids are in `reviews.parquet`.

## Make

No command in the repo makes this file. It is a saved one-off output, like `wd_reviews.parquet`.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_wd`).
