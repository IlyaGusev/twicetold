# data/other/copernicus_reviews.parquet

Copernicus (EGU open-discussion journals and EGUsphere): one row per first-round review. 8,063 reviews of 3,700 papers.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`, `review_time`, `n_chars`. `text` is the review body; `rating` is a source-specific integer (see below) and `rating_raw`
the source label; `n_chars` is the text length.

| Year | Papers | Reviews |
|---|---|---|
| 2018 | 300 | 673 |
| 2019 | 300 | 664 |
| 2020 | 300 | 653 |
| 2021 | 300 | 649 |
| 2022 | 500 | 1,092 |
| 2023 | 500 | 1,077 |
| 2024 | 500 | 1,099 |
| 2025 | 500 | 1,082 |
| 2026 | 500 | 1,074 |

## How we got it

`uv run python -I src/fetch.py copernicus`. Raw HTTP responses are cached under `RAW_JOURNAL_REVIEWS_DIR`
(`/home/yallen/data/peer-review/other`); a cached file is never downloaded again.

- It lists all Crossref peer-review records under prefix 10.5194 created from 2018-01-01
  (`https://api.crossref.org/prefixes/10.5194/works`) and keeps DOIs that end in `-rc<n>` (referee comments).
- It groups them by manuscript and keeps manuscripts with 2 or more referee comments. For each year it takes up to
  1.2 times the cap from a shuffled list (seed 0).
- It scrapes the manuscript's discussion page. When the HTML text of a comment is shorter than 300 characters, it
  adds the text of the attached PDF (`pypdf`) and removes manuscript pages.
- `rating` comes from the Crossref recommendation: accept = 4, minor revision = 3, major revision = 2, reject = 1.
- `journal` is the final journal for EGUsphere preprints when the page links one; otherwise the prefix.

`finalize` in `src/fetch.py` then applies the same rules to every source:

1. It drops reviews with fewer than 100 characters of text and duplicate (paper, text) pairs.
2. A paper's year is the year of its first review.
3. It keeps papers with 2 or more reviews and a year in 2018–2026.
4. It sorts the paper ids of each year, shuffles them with `random.Random(0)` and keeps the first 500 for 2022 and later, 300 before 2022.

Raw cache: `copernicus/crossref/`, `copernicus/html/`, `copernicus/pdf/` (10,903 files, 2026-10-29 12:52–13:31 UTC).

Last write: 2026-11-01 17:34–17:50 UTC, step `src/fetch.py copernicus` of `regen.sh` in Claude Code session `881519d6`
(log `logs/regen_gpt-6-luna-high.log`). Most responses came from the cache made on 2026-10-29.

## Reproduce

```
uv run python -I src/fetch.py copernicus --out <dir>
```

With the raw cache in place this needs no network for cached items. Without the cache it downloads again, and the
sources can have changed.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw cache into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`). `copernicus_papers.parquet` and `copernicus_stats.json` from the same run are also identical.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_reviews.parquet`.
