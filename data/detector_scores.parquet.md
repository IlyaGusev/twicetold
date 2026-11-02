# data/detector_scores.parquet

Scores of our style detector for each review of `sample_reviews.parquet`. 116,847 rows, one per review, in the same
order as `sample_reviews.parquet` (ICLR 96,995; NeurIPS 19,852).

| Column | Type | Meaning |
|---|---|---|
| `review_id` | string | Key into `sample_reviews.parquet` |
| `p_llm` | float32 | Detector probability that the full text is LLM text |
| `llm_flag` | bool | `p_llm >= q99`; q99 = 99th percentile of `p_llm` over ICLR 2017–2022 reviews (0.0875) |
| `llm_grade` | int | 0: `p_llm < q90` (q90 = 0.0150); 1: `< q99`; 2: `<= 0.95`; 3: above 0.95 |
| `p_llm_ss` | float | The same detector on the `[SUMMARY]` and `[STRENGTHS]` fields only; NaN when these fields have 200 characters or fewer (26,346 rows) |
| `llm_flag_ss` | bool | `p_llm_ss >= q99`, with the full-text q99; False where `p_llm_ss` is NaN |

## Make

```
uv run python -I src/detector.py scores --out <dir>/detector_scores.parquet --by_cell <dir>/detector_by_cell.csv
```

The detector is a logistic regression (`C=4.0`, `max_iter=3000`) on TF-IDF (sublinear tf) of hashed word 1–2-grams
and character 3–4-grams (`char_wb`), 2^18 features each, after removal of the seven ML section tags. It trains on
`intel_reviews.parquet` years 2021–2023 and tests on 2024 (AUC 0.9939). Then it scores `sample_reviews.parquet` and
sets the thresholds from ICLR 2017–2022 rows. The command also writes `detector_by_cell.csv`, the same as
`reports/detector_by_cell.csv`. CPU only, about 12 minutes. Deterministic, no network, no paid API. No chain script
runs it.

## Notes

- Git LFS stores it.

## Read by

`src/validate.py` (`rq1_evidence`, `detector_placebo`), `src/figures.py` (`likeness`, `ai`, `by_year`, `numbers`).
