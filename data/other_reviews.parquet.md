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

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/corpus.py other
```

It reads `data/other/<source>_reviews.parquet` and `<source>_papers.parquet` for `scipost`, `f1000`, `bmc`, `plos`,
`copernicus` (`--src` changes the folder). It unifies PLoS journal names, drops duplicate (paper, text) pairs and
keeps papers with 2 or more reviews. It writes this file and `reports/other_inventory.txt`.
It is deterministic and needs no network.

## Notes

Git does not track this file. `scripts/post.sh` runs `scripts/inputs.sh` when the file is missing.

## Read by

`src/panels.py other`, `src/outcomes.py` (`SOURCE_FILES`, tag `_other`), `src/validate.py` (`review_chars`),
`src/ai.py` (`reference_sample`), `src/meld.py input`, `src/figures.py` (`by_year`).
