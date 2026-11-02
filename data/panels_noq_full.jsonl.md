# data/panels_noq_full.jsonl

ICLR panels without the questions field for the sample papers of 2024–2026 that are not in `panels_noq.jsonl`.
10,800 panels with 42,265 reviews: 2024: 2,600; 2025: 2,600; 2026: 5,600. Keys: `panel_id` (`noq_<paper_id>`) and
`review_ids`. The texts are in `noq_full_reviews.parquet`.

Panel sizes: 3 reviews (2,062), 4 (7,611), 5 (1,127). With `panels_noq.jsonl`, the two files cover all 12,000
ICLR 2024–2026 sample papers.

## How we got it

1. `uv run python -I src/panels.py robust` (`src/panels.py`, functions `robust` and `robust_full`) reads
   `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl`. One run writes
   `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl`, `noq_reviews.parquet` (seed 1), then
   `panels_sub3_full.jsonl`, `noq_full_reviews.parquet` and `panels_noq_full.jsonl` (seed 11).
2. Function `robust_full`: every ICLR 2024–2026 sample paper without a panel in `panels_noq.jsonl`. It groups the
   reviews by paper (`by_paper_panels`, sorted by paper id).
3. Last write: 2026-11-01 17:28:02 UTC. Step `src/panels.py robust` of `regen.sh` in Claude Code session
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

`scripts/extract_all.sh` (pass-2 input, tag `_noq`), `src/outcomes.py` (tag `_noq`, through the clusters file).
