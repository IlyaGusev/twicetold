# data/panels_noq.jsonl

ICLR panels for the reviews without the questions field. 1,200 panels with 4,707 reviews: 400 papers each in 2024,
2025 and 2026. Keys: `panel_id` (`noq_<paper_id>`) and `review_ids`. The texts are in `noq_reviews.parquet`.

Panel sizes: 3 reviews (232), 4 (829), 5 (139).

## How we got it

1. `uv run python -I src/panels.py robust` (`src/panels.py`, functions `robust` and `robust_full`) reads
   `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl`. One run writes
   `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl`, `noq_reviews.parquet` (seed 1), then
   `panels_sub3_full.jsonl`, `noq_full_reviews.parquet` and `panels_noq_full.jsonl` (seed 11).
2. For each of ICLR 2024, 2025 and 2026, pick 400 sample papers at random. Take all their sample reviews and group
   them by paper (`by_paper_panels`, sorted by paper id).
3. Last write: 2026-11-01 17:28:00 UTC. Step `src/panels.py robust` of `regen.sh` in Claude Code session
   `881519d6` (project `~/ideas`), started at 17:27:54 UTC. The log is `logs/regen_gpt-6-luna-high.log`. It printed
   `noq reviews 4707`, `new sub3 panels 6740` and `new noq reviews 42265 papers 10800`.

## Reproduce

```
uv run python -I src/panels.py robust --out <dir>
```

The inputs (`sample_reviews.parquet`, `paper_areas.parquet`, `panels.jsonl`) come from `data/`. Use `--data <dir>` to
read them from a different folder.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the current inputs in `data/` into a
scratch folder. All seven output files of the run are byte-identical to the files in `data/`.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_noq`), `src/panels.py` (`robust_full`: skips these papers),
`src/outcomes.py` (tag `_noq`, through the clusters file).
