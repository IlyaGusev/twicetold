# data/panels_noq.jsonl

ICLR panels for the reviews without the questions section. 12,000 panels with 46,972 reviews: every sampled ICLR
paper of 2024 (3,000), 2025 (3,000) and 2026 (6,000). Keys: `panel_id` (`noq_<paper_id>`) and `review_ids`
(`noq_<OpenReview note id>`). The texts are in `noq_reviews.parquet`, with the same review ids.

Panel sizes: 3 reviews (2,294), 4 (8,440), 5 (1,266).

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/panels.py robust
```

It reads `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl` (`--data` changes the
folder). One run writes `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl` and
`noq_reviews.parquet`. It is deterministic and needs no network.

The first 1,200 panels are 400 random papers per year (seed 1). The rest are all other sampled papers of those
years. Each part groups the reviews by paper, sorted by paper id.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_noq`).
