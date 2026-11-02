# data/panels_placebo.jsonl

Cross-paper control panels for the main sample. Each panel holds 4 reviews of 4 different papers of one venue and
year. 4,500 panels: 300 for each of the 15 venue-year cells. Keys: `panel_id` (`placebo_<conf>_<year>_<k>`) and
`review_ids`.

The panels hold 18,000 review ids, 16,042 of them unique.

## How we got it

1. The same run as `sample_reviews.parquet` writes this file: `uv run python -I src/panels.py ml --data
   data/backup_predup --current_corpus data` (`src/panels.py`, function `ml`, inner function `placebo`).
2. For each (conf, year) cell of the sample, 300 times: pick 4 papers at random, then 1 review of each paper at
   random. The random generator continues from the sample draw (seed 2026).
3. Last write: 2026-11-01 17:27:54 UTC, step `src/panels.py ml` of `regen.sh` in Claude Code session `881519d6`
   (log `logs/regen_gpt-6-luna-high.log`).

The generator state at the placebo step depends on the draws from `data/backup_predup/papers.parquet` and
`reviews.parquet`. That folder was deleted on 2026-11-02 and no copy exists. So we cannot draw these panels again.

## Reproduce

Not possible (see `sample_reviews.parquet.md`). The command was:

```
uv run python -I src/panels.py ml --data data/backup_predup --current_corpus data
```

## Check

A rebuild is not possible. On 2026-11-02 we ran these consistency checks:

- 300 panels in each of the 15 cells; every panel has 4 reviews.
- In every panel, the 4 reviews come from 4 different papers.
- All 18,000 review ids are in `sample_reviews.parquet`. In the first 500 panels, all reviews of a panel are in one
  cell.
- 17,812 of the 18,000 ids are in the current `reviews.parquet`. `src/outcomes.py` drops the other ids from a panel
  and keeps the panel if 2 or more reviews stay.

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain), `src/gold.py` (`sample`: the cross-paper panels of the
reference set), `src/outcomes.py` (through the clusters file; `placebo_` prefix).
