# data/panels_sub3.jsonl

Three-review subsets of main ICLR panels. Each panel holds 3 of the reviews of one paper with 4 or more reviews.
2,773 panels with 8,319 reviews. Keys: `panel_id` (`sub3_<paper_id>`) and `review_ids`.

Panels per year: 2017: 85; 2018: 17; 2019: 115; 2020: 156; 2021–2026: 400 each.

## How we got it

1. `uv run python -I src/panels.py robust` (`src/panels.py`, functions `robust` and `robust_full`) reads
   `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl`. One run writes
   `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl`, `noq_reviews.parquet` (seed 1), then
   `panels_sub3_full.jsonl`, `noq_full_reviews.parquet` and `panels_noq_full.jsonl` (seed 11).
2. Inner function `sub3`: for each ICLR year, pick up to 400 sample papers with 4 or more reviews, then 3 of
   their reviews at random (`sub3_panel`). It keeps a panel only if all 3 reviews are in a panel of
   `panels.jsonl`.
3. Last write: 2026-11-01 17:27:59 UTC. Step `src/panels.py robust` of `regen.sh` in Claude Code session
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

All 8,319 review ids are in the current `reviews.parquet`.

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain), `src/panels.py` (`robust_full`: skips these
papers), `src/outcomes.py` (through the clusters file; the `sub3_` prefix).
