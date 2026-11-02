# data/other/f1000_reviews.parquet

F1000Research, Wellcome Open Research and Open Research Europe: one row per first-round review. 10,505 reviews of 4,400 papers.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`, `review_time`, `n_chars`. `text` is the review body; `rating` is a source-specific integer (see below) and `rating_raw`
the source label; `n_chars` is the text length.

| Year | Papers | Reviews |
|---|---|---|
| 2018 | 450 | 1,094 |
| 2019 | 500 | 1,174 |
| 2020 | 500 | 1,159 |
| 2021 | 500 | 1,130 |
| 2022 | 450 | 1,003 |
| 2023 | 500 | 1,202 |
| 2024 | 500 | 1,235 |
| 2025 | 500 | 1,264 |
| 2026 | 500 | 1,244 |

## How we got it

`uv run python -I src/fetch.py f1000`. Raw HTTP responses are cached under `RAW_JOURNAL_REVIEWS_DIR`
(`/home/yallen/data/peer-review/other`); a cached file is never downloaded again.

- For each year 2018–2026 it searches Europe PMC
  (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`) for the three journals and downloads the full-text
  XML of each hit (`/<pmcid>/fullTextXML`).
- It takes reviewer reports on version 1 of each article only.
- `text` is the report body without references, yes/no checklist answers, the reviewer expertise line and the
  confirmation statement.
- `rating`: Approved = 3, Approved with Reservations = 2, Not Approved = 1, from the recommendation field, the
  notes or the confirmation statement. `rating_raw` keeps the label.
- Candidates are scanned in shuffled order (seed 0) in batches of 100 until 500 papers with 2 or more reviews are
  found for the year.

`finalize` in `src/fetch.py` then applies the same rules to every source:

1. It drops reviews with fewer than 100 characters of text and duplicate (paper, text) pairs.
2. A paper's year is the year of its first review.
3. It keeps papers with 2 or more reviews and a year in 2018–2026.
4. It sorts the paper ids of each year, shuffles them with `random.Random(0)` and keeps the first 500.

Raw cache: `f1000/search_<year>.json` and `f1000/<pmcid>.xml` (5,909 files, 2026-10-29 12:46 to 2026-11-01 17:32 UTC).

Last write: 2026-11-01 17:28–17:32 UTC, step `src/fetch.py f1000` of `regen.sh` in Claude Code session `881519d6`
(log `logs/regen_gpt-6-luna-high.log`). Most responses came from the cache made on 2026-10-29.

## Reproduce

```
uv run python -I src/fetch.py f1000 --out <dir>
```

With the raw cache in place this needs no network for cached items. Without the cache it downloads again, and the
sources can have changed.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw cache into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`). `f1000_papers.parquet` and `f1000_stats.json` from the same run are also identical.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_reviews.parquet`.
