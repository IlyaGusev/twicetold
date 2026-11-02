# data/sample_reviews.parquet

The main ICLR and NeurIPS review sample. It is the pass-1 input of the main chain. One row per review: 116,847
reviews of 30,937 papers.

| Column | Meaning |
|---|---|
| `conf`, `year` | Venue (`ICLR`, `NeurIPS`) and conference year |
| `paper_id`, `review_id` | OpenReview forum id and note id |
| `text` | Review text, as in `reviews.parquet` |

Rows are sorted by `conf`, `year`, `paper_id`, `review_id`. Each paper has 3 to 5 reviews.

| Year | ICLR papers | ICLR reviews | NeurIPS papers | NeurIPS reviews |
|---|---|---|---|---|
| 2017 | 480 | 1,735 | – | – |
| 2018 | 918 | 2,771 | – | – |
| 2019 | 1,412 | 4,493 | – | – |
| 2020 | 2,127 | 6,546 | – | – |
| 2021 | 2,594 | 10,019 | 1,000 | 3,852 |
| 2022 | 2,617 | 10,188 | 1,000 | 3,662 |
| 2023 | 3,789 | 14,271 | 1,000 | 4,412 |
| 2024 | 3,000 | 11,596 | 1,000 | 3,895 |
| 2025 | 3,000 | 12,046 | 1,000 | 4,031 |
| 2026 | 6,000 | 23,330 | – | – |

## How we got it

1. On 2026-10-29 at 12:19 UTC, Claude Code session `881519d6` (project `~/ideas`) copied the `reviews.parquet` and
   `papers.parquet` of that time into `data/backup_predup/`. The next command in that session removed duplicate
   (paper, text) pairs from `data/reviews.parquet`; the copies kept them.
2. `uv run python -I src/panels.py ml --data data/backup_predup --current_corpus data` (`src/panels.py`, function
   `ml`, seed 2026) drew the sample from the copies in `data/backup_predup/`:
   - eligible papers: not withdrawn, 3 or more reviews, decision not `desk_reject`;
   - papers per cell: all eligible papers, except ICLR 2024 and 2025 (3,000), ICLR 2026 (6,000) and NeurIPS
     2021–2025 (1,000 each), drawn at random without replacement;
   - at most 5 reviews per paper, drawn at random;
   - text from `data/backup_predup/reviews.parquet`.
3. Last write: 2026-11-01 17:27:48 UTC, step `src/panels.py ml` of `regen.sh` in session `881519d6`. The log is
   `logs/regen_gpt-6-luna-high.log` (step start 17:27:35). Its table of papers and reviews per cell matches the
   table above.

`data/backup_predup/` was deleted on 2026-11-02 and no copy exists. `panels.py ml` cannot draw this sample again.
Treat this file as the fixed sample.

## Reproduce

Not possible. The input `data/backup_predup/reviews.parquet` and `papers.parquet` no longer exist. The command was:

```
uv run python -I src/panels.py ml --data data/backup_predup --current_corpus data
```

## Check

A rebuild is not possible. On 2026-11-02 we ran these consistency checks against the current `reviews.parquet`:

- 116,847 rows, no duplicate review ids, 3 to 5 reviews per paper, rows in sorted order.
- Papers and reviews per cell match the `regen_gpt-6-luna-high.log` table.
- 116,433 reviews (99.65%) are in the current `reviews.parquet`. All 116,433 have the same text, paper, venue and
  year there.
- 414 reviews are not in the current `reviews.parquet`: ICLR 2017 (231), ICLR 2019 (180), ICLR 2020 (3). The
  duplicate (paper, text) filter of `corpus.py ml` removed them.

## Read by

`scripts/extract_all.sh` (pass-1 input of the main chain), `src/panels.py` (`robust`), `src/between.py`
(`embed_reviews`, `load_reviews`), `src/ai.py` (`style`), `src/gold.py` (`sample`).
