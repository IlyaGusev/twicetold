# data/panels_other.jsonl

Journal panels: one panel per paper of `other_pass1.parquet`. 17,520 panels with 40,992 reviews. Keys: `panel_id`
(`other_<paper_id>`) and `review_ids`.

Panel sizes: 2 reviews (12,637), 3 (3,984), 4 (729), 5 (170). All review ids are in `other_reviews.parquet`.

## How we got it

1. `uv run python -I src/panels.py other` (`src/panels.py`, function `other`, seed 2027) reads
   `other_reviews.parquet`. One run writes `other_pass1.parquet`, `panels_other.jsonl`, `panels_other_placebo.jsonl`
   and `reports/other_placebo_preprints.csv`.
2. It keeps at most 5 reviews per paper, drawn at random (`cap_reviews_per_paper`), and sorts by `conf`, `year`,
   `paper_id`, `review_id`. This drops 77 of the 41,069 reviews.
3. `by_paper_panels` groups the capped reviews by paper.
4. Last write: 2026-11-01 17:50:10 UTC. Step `src/panels.py other` of `regen.sh` in Claude Code
   session `881519d6` (project `~/ideas`), started at 17:50:06 UTC. The log is `logs/regen_gpt-6-luna-high.log`.

## Reproduce

```
uv run python -I src/panels.py other --out <dir> --reports <dir>
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the current `other_reviews.parquet` into a
scratch folder. All three data files and `reports/other_placebo_preprints.csv` are byte-identical to the current
files.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_other`), `src/outcomes.py` (`SOURCE_PANEL_FILES`).
