# data/cocodet_main.jsonl

CoCoDet class probabilities for the 26,484 ICLR and NeurIPS reviews of `cocodet_input.parquet`. One JSON object per
line: `review_id` (string), `human`, `mix`, `ai` (floats that sum to 1). Largest class: human 24,335; mix 1,136;
ai 1,013. A review counts as AI or mixed when `human` is not the largest of the three.

## Make

```
uv run python -I src/cocodet.py scores --input data/cocodet_input.parquet --out <dir>/cocodet_main.jsonl
```

It runs CoCoDet (`settings.COCODET_REPO`, file `settings.COCODET_WEIGHTS_FILE`, base model `settings.COCODET_BASE_MODEL`) on CPU: 512 tokens, batch 8. It downloads the weights from Hugging Face (network); the code pins no revision. No paid API. It is CPU-heavy and no chain script runs it. It removes only the seven ML section tags (`[SUMMARY]`, `[MAIN]`, `[STRENGTHS]`, `[WEAKNESSES]`, `[QUESTIONS]`, `[LIMITATIONS]`, `[OTHER]`). It appends to `--out` and skips the ids it holds.

## Notes

- The line order is the scoring order (shuffled with `random_state=0`, then sorted by text length inside chunks of
  256 rows), not the input order.
- Git LFS stores it.

## Read by

`src/validate.py rq1_evidence`; `src/figures.py by_year` (list `COCODET`), which `figures.py table` calls.
