# data/icml_pass1.parquet

The pass-1 input for the ICML chain: columns `review_id` and `text` of `icml_sample.parquet`, in the same row
order. 11,500 rows.

## How we got it

1. `uv run python -I src/panels.py icml` (`src/panels.py`, function `icml`, seed 11) reads
   `icml2025_reviews.parquet` and `icml2026_reviews.parquet`. One run writes `icml_pass1.parquet`,
   `icml_sample.parquet` and `panels_icml.jsonl`.
2. For each year: keep reviews with more than 200 characters; drop duplicate (paper, text) pairs; pick 1,500 papers
   with 3 or more reviews at random; shuffle their reviews (`random_state=0`); keep at most 5 reviews per paper.
3. Last write: 2026-11-01 17:50:15 UTC. Step `src/panels.py icml` of `regen.sh` in Claude Code
   session `881519d6` (project `~/ideas`), started at 17:50:14 UTC. The log is `logs/regen_gpt-6-luna-high.log`.

## Reproduce

```
uv run python -I src/panels.py icml --out <dir>
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the current inputs in `data/` into a
scratch folder. All three output files are byte-identical to the files in `data/`.
`pandas.DataFrame.equals` also gives `True`. The review ids are the same as in `icml_sample.parquet`,
in the same order.

## Read by

`scripts/extract_all.sh` (pass-1 input, tag `_icml`).
