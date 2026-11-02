# data/noq_full_reviews.parquet

The reviews of `panels_noq_full.jsonl` with the questions field removed. 42,265 rows. Columns: `conf`, `year`,
`paper_id`, `review_id`, `text`.

| Year | Papers | Reviews |
|---|---|---|
| 2024 | 2,600 | 10,051 |
| 2025 | 2,600 | 10,443 |
| 2026 | 5,600 | 21,771 |

## How we got it

1. `uv run python -I src/panels.py robust` (`src/panels.py`, functions `robust` and `robust_full`) reads
   `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl`. One run writes
   `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl`, `noq_reviews.parquet` (seed 1), then
   `panels_sub3_full.jsonl`, `noq_full_reviews.parquet` and `panels_noq_full.jsonl` (seed 11).
2. Function `robust_full`: the ICLR 2024–2026 sample reviews of papers not in `panels_noq.jsonl`. `drop_questions`
   removes the text from `[QUESTIONS]` up to the next `[LIMITATIONS]` or `[OTHER]` tag, or to the end.
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
`pandas.DataFrame.equals` also gives `True`.

## Read by

`scripts/extract_all.sh` (pass-1 input, tag `_noq`).
