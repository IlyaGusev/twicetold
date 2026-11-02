# data/panels_placebo_area.jsonl

Same-area cross-paper control panels for ICLR. Each panel holds 3 reviews of 3 different ICLR papers of one year
with the same predicted primary area. 2,996 panels with 8,988 review ids (8,276 unique). Keys: `panel_id`
(`placeboA_ICLR_<year>_<k>`), `area` and `review_ids`.

Panels per year: 2017: 297; 2018: 300; 2019: 299; 2020–2026: 300 each. The panels use 21 areas.

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/panels.py robust
```

It reads `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl` (`--data` changes the
folder). One run writes `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl` and
`noq_reviews.parquet`. It is deterministic and needs no network.

Inner function `placebo_area` (seed 1): the area of a paper is `area_pred` from `paper_areas.parquet` (`other`
when missing). For each ICLR year, 300 times: pick the area of a random paper, pick 3 papers of that area, then 1
review of each paper. It skips a draw when the area has fewer than 3 papers, so 2017 and 2019 have fewer than 300
panels.

## Notes

8,872 of the 8,988 review ids are in `reviews.parquet`. The other 116 are ICLR 2017 and 2019 sample reviews that
`reviews.parquet` does not hold (see `sample_reviews.parquet.md`).

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain).
