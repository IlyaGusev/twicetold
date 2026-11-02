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

1. Detector (`src/detector.py`, not committed in this form; a copy is in
   `/tmp/claude-1000/-home-yallen-ideas/881519d6-0823-4954-a4c3-bf0b4f7ffed9/scratchpad/pre_fire/src/detector.py`,
   which is not permanent):
   - features: hashed word 1–2-grams and character 3–4-grams (`char_wb`), 2^18 features each, TF-IDF with
     sublinear tf, after removal of the seven ML section tags;
   - logistic regression, `C=4.0`, `max_iter=3000`;
   - training data: `intel_reviews.parquet` years 2021–2023 (57,692 texts); test: year 2024 (19,037 texts), AUC
     0.9939.
2. It scores every row of `sample_reviews.parquet`, sets the thresholds from ICLR 2017–2022 rows, and writes this
   file. It then prints a comparison with `gptzero.parquet`.
3. Command, Claude Code session `1e2ee420` (project `~/twicetold`), 2026-10-29 04:58:48 UTC:

   ```
   setsid nohup uv run python -I src/detector.py > logs/detector3.log 2>&1
   ```

   The log is `logs/detector3.log`. The last write was on 2026-10-29 at 05:10:50 UTC. CPU only, on this machine. No
   paid API. Two earlier runs wrote `logs/detector.log` and `logs/detector2.log`.

## Reproduce

No current command gives this exact file. The nearest is:

```
uv run python -I src/ai.py style --out <dir>/detector_scores.parquet --by_cell <dir>/detector_by_cell.csv
```

`src/ai.py style` uses the same features and model for `p_llm`, `llm_flag` and `llm_grade`. It sets `llm_flag_ss`
with a separate threshold: the 99th percentile of `p_llm_ss` over ICLR 2017–2022 (0.376 on this file), and leaves it
empty where `p_llm_ss` is NaN. On this file the two rules agree on 79.0% of the rows that have `p_llm_ss`.

## Check

On 2026-11-02 we did not rerun the detector (the run takes about 12 minutes). We checked:

- 116,847 rows; the `review_id` order equals the current `sample_reviews.parquet`.
- No NaN except `p_llm_ss` (26,346). `p_llm` is in [5.1e-6, 0.99997].
- With venue and year from the current `sample_reviews.parquet`: q90 = 0.0150, q99 = 0.0875, and `llm_flag` equals
  `p_llm >= q99` on all rows.
- `llm_flag_ss` equals `p_llm_ss >= q99` (full-text q99) on all rows. This matches the detector code of
  2026-10-29, not the current `src/ai.py style`.

## Read by

`src/validate.py` (`load_flags`, used by `rq1_evidence` and `paper_side`; `detector_placebo`), `src/figures.py`
(`likeness`, `ai`, `by_year`, `numbers`).
