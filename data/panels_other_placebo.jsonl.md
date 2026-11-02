# data/panels_other_placebo.jsonl

Same-journal cross-paper control panels for the journal sources. Each panel holds 3 reviews of 3 different papers of
one journal, source and year. 6,450 panels (150 for each of 43 source-year cells) with 19,350 review ids (14,783
unique). Keys: `panel_id` (`otherplacebo_<conf>_<year>_<k>`) and `review_ids`.

Panels per source: BMC-Med 1,200; Copernicus 1,350; F1000 1,350; PLOS 1,200; SciPost 1,350. All review ids are in
`other_reviews.parquet`.

## How we got it

1. `uv run python -I src/panels.py other` (`src/panels.py`, function `other`, seed 2027) reads
   `other_reviews.parquet`. One run writes `other_pass1.parquet`, `panels_other.jsonl`, `panels_other_placebo.jsonl`
   and `reports/other_placebo_preprints.csv`.
2. It keeps at most 5 reviews per paper, drawn at random (`cap_reviews_per_paper`), and sorts by `conf`, `year`,
   `paper_id`, `review_id`. This drops 77 of the 41,069 reviews.
3. Inner function `placebo`: in each (source, year) cell, take the journals with 3 or more papers, except the
   Copernicus preprint server `EGUsphere`. 150 times: pick a journal with weight equal to its paper count, pick 3
   of its papers, then 1 review of each paper. The run writes the preprint share per cell to
   `reports/other_placebo_preprints.csv`.
4. Last write: 2026-11-01 17:50:14 UTC. Step `src/panels.py other` of `regen.sh` in Claude Code
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

`scripts/extract_all.sh` (pass-2 input, tag `_other`), `src/outcomes.py` (`SOURCE_PANEL_FILES`; the
`otherplacebo_` prefix marks a control panel).
