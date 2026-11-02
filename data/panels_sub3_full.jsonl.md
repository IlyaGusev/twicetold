# data/panels_sub3_full.jsonl

More three-review subsets for ICLR 2023 and 2026. Together with `panels_sub3.jsonl`, they cover every sample paper
of those years with 4 or more reviews. 6,740 panels with 20,220 reviews: 2023: 2,222; 2026: 4,518. Keys:
`panel_id` (`sub3_<paper_id>`) and `review_ids`.

## How we got it

1. `uv run python -I src/panels.py robust` (`src/panels.py`, functions `robust` and `robust_full`) reads
   `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl`. One run writes
   `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl`, `noq_reviews.parquet` (seed 1), then
   `panels_sub3_full.jsonl`, `noq_full_reviews.parquet` and `panels_noq_full.jsonl` (seed 11).
2. Function `robust_full`: for ICLR 2023 and 2026, every sample paper with 4 or more reviews that has no panel in
   `panels_sub3.jsonl` gets 3 reviews at random. It keeps a panel only if all 3 reviews are in a panel of
   `panels.jsonl`.
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

All 20,220 review ids are in the current `reviews.parquet`.

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain), `src/outcomes.py` (through the clusters file; the
`sub3_` prefix).
