# data/icml_sample.parquet

The ICML 2025 and 2026 review sample. One row per review: 11,500 rows. Columns: `conf` (`ICML`), `year`,
`paper_id`, `review_id`, `rating`, `confidence`, `text`. Rows are in sample order (shuffled within a year), not
sorted.

| Year | Papers | Reviews | Median characters |
|---|---|---|---|
| 2025 | 1,500 | 5,713 | 3,526 |
| 2026 | 1,500 | 5,787 | 3,003 |

## Make

```
uv run python -I src/panels.py icml
```

`scripts/inputs.sh` runs this step. The same step writes `panels_icml.jsonl`. Deterministic (seed 11), no API. It
reads `icml2025_reviews.parquet` and `icml2026_reviews.parquet`. For each year it keeps reviews with more than 200
characters, drops duplicate (paper, text) pairs, picks 1,500 random papers with 3 or more reviews, shuffles their
reviews (`random_state=0`) and keeps at most 5 reviews per paper.

## Notes

- Git LFS stores it.

## Read by

- `scripts/extract_all.sh`: pass-1 input, tag `_icml`.
- `src/outcomes.py` (`SOURCE_FILES`): `panels` (tag `_icml`) and `pairs`.
- `src/validate.py review_chars` (through `SOURCE_FILES`).
- `src/meld.py input`.
