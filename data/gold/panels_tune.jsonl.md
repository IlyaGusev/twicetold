# data/gold/panels_tune.jsonl

The 60 panels of the reference set. One line per panel: `panel_id`, `review_ids`, `kind`, `year`, `split`.

| Kind | Meaning | dev | test |
|---|---|---|---|
| `same` | 4 reviews of one ICLR paper (from `blind_key.json`), 10 per year | 28 | 12 |
| `cross` | 4 reviews of 4 different ICLR papers of one year (from `panels_placebo.jsonl`), 5 per year | 12 | 8 |

Years: 2018, 2022, 2024, 2026.

## Make

```
uv run python -I src/gold.py sample --out <dir>
```

Step 1 of `scripts/gold.sh <dir>`. The same step writes `tune_reviews.parquet`. Deterministic (seed 7), no API. It
reads `data/gold/blind_key.json`, `data/panels_placebo.jsonl` and `data/sample_reviews.parquet`.

1. Same-paper panels: every entry of `blind_key.json`, in sorted code order.
2. Cross-paper panels: for each year, one `numpy.random.default_rng(7)` picks 5 of the sorted
   `placebo_ICLR_<year>_*` panels.
3. Split: in each (kind, year) cell, in panel order, the last 3 same-paper panels or the last 2 cross-paper panels
   are `test`; the rest are `dev`.

`tests/test_gold.py` rebuilds the file and checks that it is byte-identical.

## Read by

`src/gold.py claim_inputs`, `council_groups` (as the `src/extract.py pass2` input) and `group_inputs`.
