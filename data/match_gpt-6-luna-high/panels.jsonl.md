# data/match_gpt-6-luna-high/panels.jsonl

One two-review panel per reference review: 240 lines, `panel_id` = review id, `review_ids` =
`["ref_<review_id>", "<review_id>"]`. Claim id `R0-k` is reference claim k; `R1-k` is Luna claim k.

## Make

`src/gold.py match_claims` (see `claims.jsonl.md`). Deterministic, no API.

## Notes

- Git LFS stores it.

## Read by

`src/extract.py pass2`, called by `src/gold.py match_claims`.
