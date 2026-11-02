# data/panels_icml.jsonl

ICML 2025 and 2026 panels: one panel per sampled paper. 3,000 panels with 11,500 reviews. Keys: `panel_id`
(`icml_<paper_id>`) and `review_ids`. Sorted by paper id.

Panel sizes: 3 reviews (576), 4 (2,348), 5 (76). All review ids are in `icml_sample.parquet`.

## How we got it

1. `uv run python -I src/panels.py icml` (`src/panels.py`, function `icml`, seed 11) reads
   `icml2025_reviews.parquet` and `icml2026_reviews.parquet`. One run writes `icml_pass1.parquet`,
   `icml_sample.parquet` and `panels_icml.jsonl`.
2. For each year: keep reviews with more than 200 characters; drop duplicate (paper, text) pairs; pick 1,500 papers
   with 3 or more reviews at random; shuffle their reviews (`random_state=0`); keep at most 5 reviews per paper.
3. `by_paper_panels` groups the sample by paper.
4. Last write: 2026-11-01 17:50:16 UTC. Step `src/panels.py icml` of `regen.sh` in Claude Code
   session `881519d6` (project `~/ideas`), started at 17:50:14 UTC. The log is `logs/regen_gpt-6-luna-high.log`.

## Reproduce

```
uv run python -I src/panels.py icml --out <dir>
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the current inputs in `data/` into a
scratch folder. All three output files are byte-identical to the files in `data/`.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_icml`), `src/outcomes.py` (`PANEL_FILES`, `SOURCE_PANEL_FILES`).
