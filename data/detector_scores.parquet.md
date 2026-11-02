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

## How we got it

1. Detector: `src/detector.py` of that time (in git at `c57b0a8^`). `src/detector.py scores` is the same code
   without comments, with paths from `src/settings.py` and without the GPTZero comparison print:
   - features: hashed word 1–2-grams and character 3–4-grams (`char_wb`), 2^18 features each, TF-IDF with
     sublinear tf, after removal of the seven ML section tags;
   - logistic regression, `C=4.0`, `max_iter=3000`;
   - training data: `intel_reviews.parquet` years 2021–2023 (57,692 texts); test: year 2024 (19,037 texts), AUC
     0.9939.
2. It scores every row of `sample_reviews.parquet`, sets the thresholds from ICLR 2017–2022 rows, and writes this
   file.
3. Command, Claude Code session `1e2ee420` (project `~/twicetold`), 2026-10-29 04:58:48 UTC:

   ```
   setsid nohup uv run python -I src/detector.py > logs/detector3.log 2>&1
   ```

   The log is `logs/detector3.log`. The last write was on 2026-10-29 at 05:10:50 UTC. CPU only, on this machine. No
   paid API. Two earlier runs wrote `logs/detector.log` and `logs/detector2.log`.

## Reproduce

```
uv run python -I src/detector.py scores --out <dir>/detector_scores.parquet --by_cell <dir>/detector_by_cell.csv
```

CPU only, about 12 minutes.

## Check

On 2026-11-02 we ran the "Reproduce" command into a scratch folder (CPU, about 12 minutes). The output is identical
to this file (`pandas.DataFrame.equals`), and `detector_by_cell.csv` is identical to `reports/detector_by_cell.csv`.
The run printed the same test AUC (0.9939) and thresholds (q90 = 0.015, q99 = 0.087).

## Read by

`src/validate.py` (`rq1_evidence`, `detector_placebo`), `src/figures.py`
(`likeness`, `ai`, `by_year`, `numbers`).
