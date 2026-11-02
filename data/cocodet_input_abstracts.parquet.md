# data/cocodet_input_abstracts.parquet

The title and abstract of ICLR and NeurIPS papers, for CoCoDet. One row per paper: 29,518 rows. Columns:

| Column | Meaning |
|---|---|
| `review_id` | The OpenReview paper id. The name is `review_id` because `src/ai.py cocodet` reads that column |
| `conf`, `year` | Venue and conference year |
| `text` | `title + "\n\n" + abstract` from `papers.parquet` (missing parts as empty strings) |

Rows per venue and year:

| Year | ICLR | NeurIPS |
|---|---|---|
| 2019 | 1,406 | – |
| 2020 | 2,120 | – |
| 2021 | 2,591 | 1,000 |
| 2022 | 2,616 | 1,000 |
| 2023 | 3,787 | 1,000 |
| 2024 | 3,000 | 999 |
| 2025 | 2,999 | 1,000 |
| 2026 | 6,000 | – |

## How we got it

No script in the repo makes this file.

On 2026-10-30 at 22:08:06 UTC, an inline Python command in Claude Code session `881519d6` (project `~/ideas`) wrote
it:

1. It read the distinct (`panel_id`, `conf`, `year`) rows of `data/pairs_gpt-6-luna.parquet` and kept `year >= 2019`.
2. It joined `papers.parquet` on `conf` and paper id.
3. It wrote `review_id = paper_id`, `conf`, `year` and `text`.

`pairs_gpt-6-luna.parquet` no longer exists.

## Reproduce

We cannot reproduce this file: the pair table of step 1 is gone.

## Check

On 2026-11-02 we compared this file with the current tables:

- 29,518 rows, no duplicate ids.
- All 29,518 ids are in the current `papers.parquet`, and all texts equal `title + "\n\n" + abstract` of the
  current table.
- All 29,518 ids are panels in the current `panels.jsonl`.

## Read by

`src/ai.py cocodet` (input). `src/validate.py paper_side` reads only the output `cocodet_abstracts.jsonl`.
