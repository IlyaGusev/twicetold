# data/other/plos_reviews.parquet

PLOS journals (ONE, Biology, Computational Biology, Genetics, Medicine, Pathogens, NTD, Global Public Health, Digital Health): one row per first-round review. 9,015 reviews of 3,756 papers.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`, `review_time`, `n_chars`. `text` is the review body; `rating` is a source-specific integer (see below) and `rating_raw`
the source label; `n_chars` is the text length.

| Year | Papers | Reviews |
|---|---|---|
| 2019 | 500 | 1,174 |
| 2020 | 500 | 1,206 |
| 2021 | 500 | 1,181 |
| 2022 | 500 | 1,176 |
| 2023 | 500 | 1,208 |
| 2024 | 500 | 1,191 |
| 2025 | 500 | 1,242 |
| 2026 | 256 | 637 |

## How we got it

`uv run python -I src/fetch.py plos`. Raw HTTP responses are cached under `RAW_JOURNAL_REVIEWS_DIR`
(`/home/yallen/data/peer-review/other`); a cached file is never downloaded again.

- For each year 2018–2026 it searches Europe PMC for the nine journals with the phrase `"Decision Letter 0"`,
  then downloads the full-text XML.
- It skips papers that are resubmissions and papers whose earlier reviews appear in an author response.
- It takes the first decision letter that contains `Reviewer #<n>:` and splits it into one text per reviewer.
  It removes the editor's text and the journal boilerplate.
- Short yes/no answers to the checklist go to `rating_raw`. `rating` uses the "technically sound" question:
  Yes = 3, Partly = 2, No = 1.
- `field` is `multidisciplinary` for PLOS ONE and `biomed` for the others.

`finalize` in `src/fetch.py` then applies the same rules to every source:

1. It drops reviews with fewer than 100 characters of text and duplicate (paper, text) pairs.
2. A paper's year is the year of its first review.
3. It keeps papers with 2 or more reviews and a year in 2018–2026.
4. It sorts the paper ids of each year, shuffles them with `random.Random(0)` and keeps the first 500.

Raw cache: `plos/search_<year>.json` and `plos/<pmcid>.xml` (5,309 files, 2026-10-29 13:22 to 2026-11-01 17:33 UTC).

Last write: 2026-11-01 17:33 UTC, step `src/fetch.py plos` of `regen.sh` in Claude Code session `881519d6`
(log `logs/regen_gpt-6-luna-high.log`). Most responses came from the cache made on 2026-10-29.

## Reproduce

```
uv run python -I src/fetch.py plos --out <dir>
```

With the raw cache in place this needs no network for cached items. Without the cache it downloads again, and the
sources can have changed.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw cache into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`). `plos_papers.parquet` and `plos_stats.json` from the same run are also identical.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_reviews.parquet`.
