# data/other_pass1.parquet

The pass-1 input for the journal chain. Columns: `review_id`, `text`. 40,992 reviews of 17,520 papers.

| Source | Papers | Reviews |
|---|---|---|
| BMC-Med | 2,612 | 6,652 |
| Copernicus | 3,700 | 8,055 |
| F1000 | 4,400 | 10,496 |
| PLOS | 3,756 | 8,970 |
| SciPost | 3,052 | 6,819 |

## How we got it

1. `uv run python -I src/panels.py other` (`src/panels.py`, function `other`, seed 2027) reads
   `other_reviews.parquet`. One run writes `other_pass1.parquet`, `panels_other.jsonl`, `panels_other_placebo.jsonl`
   and `reports/other_placebo_preprints.csv`.
2. It keeps at most 5 reviews per paper, drawn at random (`cap_reviews_per_paper`), and sorts by `conf`, `year`,
   `paper_id`, `review_id`. This drops 77 of the 41,069 reviews.
3. Last write: 2026-11-01 17:50:08 UTC. Step `src/panels.py other` of `regen.sh` in Claude Code
   session `881519d6` (project `~/ideas`), started at 17:50:06 UTC. The log is `logs/regen_gpt-6-luna-high.log`.

## Reproduce

```
uv run python -I src/panels.py other --out <dir> --reports <dir>
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the current `other_reviews.parquet` into a
scratch folder. All three data files and `reports/other_placebo_preprints.csv` are byte-identical to the current
files. `pandas.DataFrame.equals` also gives `True`.

## Read by

`scripts/extract_all.sh` (pass-1 input, tag `_other`, `--paper_kind general`).
