# data/panels_sub3.jsonl

Three-review subsets of main ICLR panels. Each panel holds 3 of the reviews of one paper with 4 or more reviews.
9,513 panels with 28,539 reviews. Keys: `panel_id` (`sub3_<paper_id>`) and `review_ids`.

Panels per year: 2017: 85; 2018: 17; 2019: 115; 2020: 156; 2021, 2022, 2024, 2025: 400 each; 2023: 2,622;
2026: 4,918. All review ids are in `reviews.parquet`.

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/panels.py robust
```

It reads `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl` (`--data` changes the
folder). One run writes `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl` and
`noq_reviews.parquet`. It is deterministic and needs no network.

- Inner function `sub3` (seed 1): for each ICLR year, pick up to 400 sample papers with 4 or more reviews, then 3 of
  their reviews at random (`sub3_panel`).
- `full_sub3` (seed 11): for 2023 and 2026, do the same for every other sample paper with 4 or more reviews.
- Both keep a panel only if all 3 reviews are in a panel of `panels.jsonl`.

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain).
