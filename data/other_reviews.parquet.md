# data/other_reviews.parquet

Journal reviews from five open-review sources, merged. One row per first-round review: 41,069 reviews of 17,520
papers.

| Column | Meaning |
|---|---|
| `conf` | Source: `BMC-Med`, `Copernicus`, `F1000`, `PLOS`, `SciPost` |
| `field`, `journal` | Field label and journal name (PLoS names unified to `PLOS`, `PLOS ONE`) |
| `year` | Year of the paper's first review |
| `paper_id`, `review_id`, `reviewer` | Source ids; `reviewer` is a label such as `Reviewer 2` |
| `rating`, `rating_raw` | Source-specific integer and source label; see `other/<source>_reviews.parquet.md` |
| `text`, `n_chars` | Review body and its length |
| `review_time` | Review date |

Rows per source and year:

| Year | BMC-Med | Copernicus | F1000 | PLOS | SciPost |
|---|---|---|---|---|---|
| 2018 | 442 | 673 | 1,094 | 0 | 246 |
| 2019 | 454 | 664 | 1,174 | 1,174 | 301 |
| 2020 | 847 | 653 | 1,159 | 1,206 | 518 |
| 2021 | 893 | 649 | 1,130 | 1,181 | 849 |
| 2022 | 1,159 | 1,092 | 1,003 | 1,176 | 955 |
| 2023 | 1,265 | 1,077 | 1,202 | 1,208 | 915 |
| 2024 | 1,275 | 1,099 | 1,235 | 1,191 | 996 |
| 2025 | 331 | 1,082 | 1,264 | 1,242 | 1,123 |
| 2026 | 0 | 1,074 | 1,244 | 637 | 917 |

## How we got it

1. `src/fetch.py <source>` for each of `scipost`, `f1000`, `bmc`, `plos`, `copernicus` writes
   `data/other/<source>_reviews.parquet`. Each source has its own `.md` file there.
2. `uv run python -I src/corpus.py other` (`src/corpus.py`, function `other`) merges them:
   - unifies PLoS journal names;
   - drops duplicate (paper, text) pairs across sources (0 in this build);
   - keeps papers with 2 or more reviews;
   - writes this file, `other_papers.parquet` and `reports/other_inventory.txt`.

Last write: 2026-11-01 17:50 UTC, step `src/corpus.py other` of `regen.sh` in Claude Code session `881519d6`
(log `logs/regen_gpt-6-luna-high.log`).

## Reproduce

```
uv run python -I src/corpus.py other --out <dir> --report <dir>/other_inventory.txt
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw cache into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`). `reports/other_inventory.txt` is also identical.

## Read by

`src/panels.py other`, `src/outcomes.py` (tag `_other`), `src/validate.py` (`review_chars`), `src/ai.py` (`reference`, `wordshare_fields`),
`src/figures.py` (corpus table).
