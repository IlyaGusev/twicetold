# data/other/bmc_reviews.parquet

BMC Medicine: one row per first-round review. 6,666 reviews of 2,612 papers.

Columns: `conf`, `field`, `journal`, `year`, `paper_id`, `review_id`, `reviewer`, `rating`, `rating_raw`, `text`, `review_time`, `n_chars`. `text` is the review body; `rating` is a source-specific integer (see below) and `rating_raw`
the source label; `n_chars` is the text length.

| Year | Papers | Reviews |
|---|---|---|
| 2018 | 178 | 442 |
| 2019 | 193 | 454 |
| 2020 | 334 | 847 |
| 2021 | 327 | 893 |
| 2022 | 463 | 1,159 |
| 2023 | 485 | 1,265 |
| 2024 | 500 | 1,275 |
| 2025 | 132 | 331 |

## How we got it

`uv run python -I src/fetch.py bmc`. Raw HTTP responses are cached under `RAW_JOURNAL_REVIEWS_DIR`
(`/home/yallen/data/peer-review/other`); a cached file is never downloaded again.

- It downloads `https://huggingface.co/datasets/JerMa88/BMC_Medicine_Peer_Reviews/resolve/main/train.parquet`
  (one file, `main` branch, not pinned).
- It keeps first-round reports only (`review_src` ends in `_V0_R<n>.pdf`).
- `text` is the part between `Reviewer's report:` and the first form question (for example "Are the methods
  appropriate").
- The date comes from the `Date:` line, the title from the `Title:` line. There is no rating. Licence CC BY 4.0.

`finalize` in `src/fetch.py` then applies the same rules to every source:

1. It drops reviews with fewer than 100 characters of text and duplicate (paper, text) pairs.
2. A paper's year is the year of its first review.
3. It keeps papers with 2 or more reviews and a year in 2018–2026.
4. It sorts the paper ids of each year, shuffles them with `random.Random(0)` and keeps the first 500.

Raw cache: `bmc/train.parquet` (2026-10-29 12:37 UTC).

Last write: 2026-11-01 17:32 UTC, step `src/fetch.py bmc` of `regen.sh` in Claude Code session `881519d6`
(log `logs/regen_gpt-6-luna-high.log`). Most responses came from the cache made on 2026-10-29.

## Reproduce

```
uv run python -I src/fetch.py bmc --out <dir>
```

With the raw cache in place this needs no network for cached items. Without the cache it downloads again, and the
sources can have changed.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw cache into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`). `bmc_papers.parquet` and `bmc_stats.json` from the same run are also identical.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_reviews.parquet`.
