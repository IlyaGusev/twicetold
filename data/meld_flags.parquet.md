# data/meld_flags.parquet

MELD flags for every real review: the `sample`, `icml` and `journal` sets of `meld_input.parquet`. One row per review:
169,416 rows, unique `review_id`.

| Column | Meaning |
|---|---|
| `review_id`, `conf`, `year` | As in the review tables |
| `score`, `n_tokens`, `family`, `task` | From `meld_scores.jsonl` |
| `flag` | `score` above the threshold (25,085 rows) |
| `written` | `flag` and `task == "generate"` (12,007 rows) |

The threshold is the `settings.MELD_FLAG_QUANTILE` (0.999) quantile of `score` over the real reviews with
`year <= settings.MELD_CALIBRATION_LAST_YEAR` (2022): 63,958 reviews of all sources, threshold 2.39.

## Make

`scripts/post.sh` runs this step:

```
uv run python -I src/meld.py flags
```

It also writes `reports/meld_rates.csv` (flag and AI-written rates by set, source and year, including the `intel_ai`
and `rewrite` recall sets) and `reports/meld_summary.csv` (threshold, calibration size, the largest pre-2023 flag
rates, recall). Deterministic, no network, no paid API.

## Notes

- Git LFS stores it. `scripts/post.sh` rebuilds it.

## Read by

`src/validate.py` (`detector_placebo`, `rq1_evidence`, `journal_gaps`), `src/figures.py` (`ai`, `rates`, `by_year`,
`numbers`).
