# data/other/scipost_reviews.parquet

SciPost (physics and related fields): one row per first-round review. 6,820 reviews of 3,052 papers.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`, `review_time`, `n_chars`. `text` is the review body; `rating` is a source-specific integer (see below) and `rating_raw`
the source label; `n_chars` is the text length.

| Year | Papers | Reviews |
|---|---|---|
| 2018 | 110 | 246 |
| 2019 | 136 | 301 |
| 2020 | 229 | 518 |
| 2021 | 376 | 849 |
| 2022 | 433 | 955 |
| 2023 | 412 | 915 |
| 2024 | 449 | 996 |
| 2025 | 500 | 1,123 |
| 2026 | 407 | 917 |

## How we got it

`uv run python -I src/fetch.py scipost`. Raw HTTP responses are cached under `RAW_JOURNAL_REVIEWS_DIR`
(`/home/yallen/data/peer-review/other`); a cached file is never downloaded again.

- It reads every page of the SciPost API (`https://scipost.org/api/submissions/?format=json&limit=100&offset=<n>`)
  and skips resubmissions.
- It keeps reports with status `vetted` and type `report_normal`.
- `text` joins the fields strengths, weaknesses, report and requested changes with tags `[STRENGTHS]`,
  `[WEAKNESSES]`, `[REPORT]`, `[REQUESTED CHANGES]`.
- `rating` is the rounded mean of validity, significance, originality and clarity on a 1–6 scale
  (`poor`=1 … `top`=6). `rating_raw` lists all six criteria.
- `year` starts as the submission year. Every paper has licence CC BY 4.0.

`finalize` in `src/fetch.py` then applies the same rules to every source:

1. It drops reviews with fewer than 100 characters of text and duplicate (paper, text) pairs.
2. A paper's year is the year of its first review.
3. It keeps papers with 2 or more reviews and a year in 2018–2026.
4. It sorts the paper ids of each year, shuffles them with `random.Random(0)` and keeps the first 500.

Raw cache: `scipost/count.json`, `scipost/page_<offset>.json` (92 files, 2026-10-29 12:39–12:42 UTC).

Last write: 2026-11-01 17:28 UTC, step `src/fetch.py scipost` of `regen.sh` in Claude Code session `881519d6`
(log `logs/regen_gpt-6-luna-high.log`). Most responses came from the cache made on 2026-10-29.

## Reproduce

```
uv run python -I src/fetch.py scipost --out <dir>
```

With the raw cache in place this needs no network for cached items. Without the cache it downloads again, and the
sources can have changed.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw cache into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`). `scipost_papers.parquet` and `<source>_stats.json` from the same run are also identical.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_reviews.parquet`.
