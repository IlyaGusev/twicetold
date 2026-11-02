# data/icml_sample.parquet

The ICML 2025 and 2026 review sample with metadata. 11,500 rows. Columns: `conf` (`ICML`), `year`, `paper_id`,
`review_id`, `rating`, `confidence`, `text`. The rows are in sample order (shuffled within a year), not sorted.

| Year | Papers | Reviews | Median characters |
|---|---|---|---|
| 2025 | 1,500 | 5,713 | 3,526 |
| 2026 | 1,500 | 5,787 | 3,003 |

## How we got it

1. `uv run python -I src/panels.py icml` (`src/panels.py`, function `icml`, seed 11) reads
   `icml2025_reviews.parquet` and `icml2026_reviews.parquet`. One run writes `icml_pass1.parquet`,
   `icml_sample.parquet` and `panels_icml.jsonl`.
2. For each year: keep reviews with more than 200 characters; drop duplicate (paper, text) pairs; pick 1,500 papers
   with 3 or more reviews at random; shuffle their reviews (`random_state=0`); keep at most 5 reviews per paper.
3. Last write: 2026-11-01 17:50:16 UTC. Step `src/panels.py icml` of `regen.sh` in Claude Code
   session `881519d6` (project `~/ideas`), started at 17:50:14 UTC. The log is `logs/regen_gpt-6-luna-high.log`.
   The log table matches the table above.

## Reproduce

```
uv run python -I src/panels.py icml --out <dir>
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the current inputs in `data/` into a
scratch folder. All three output files are byte-identical to the files in `data/`.
`pandas.DataFrame.equals` also gives `True`.

## Read by

`src/outcomes.py` (tag `_icml`: `source_panels`, `pairs`, `review_chars`), `src/figures.py` (`REVIEW_FILES`),
`src/ai.py` (`wordshare`).
