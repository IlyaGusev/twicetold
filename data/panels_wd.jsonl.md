# data/panels_wd.jsonl

Panels of withdrawn ICLR 2026 papers: one panel per paper of `wd_reviews.parquet`. 1,000 panels with 3,891 reviews.
Keys: `panel_id` (`wd_<paper_id>`) and `review_ids`. Sorted by paper id.

Panel sizes: 3 reviews (164), 4 (781), 5 (55).

Each panel holds all reviews of one paper of `wd_reviews.parquet`. All 3,891 review ids are in `reviews.parquet`.

## Make

```
uv run python -I src/panels.py wd
```

It groups `wd_reviews.parquet` by paper. Deterministic; no network.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_wd`).
