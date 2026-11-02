# data/cocodet_icml.jsonl

CoCoDet class probabilities for the 11,500 ICML reviews of `icml_sample.parquet`. One JSON object per line:
`review_id` (string), `human`, `mix`, `ai` (floats that sum to 1). Largest class: human 10,728; mix 632; ai 140.

## Make

```
uv run python -I src/cocodet.py scores --input data/icml_sample.parquet --out <dir>/cocodet_icml.jsonl
```

It runs CoCoDet (`settings.COCODET_REPO`, file `settings.COCODET_WEIGHTS_FILE`, base model `settings.COCODET_BASE_MODEL`) on CPU: 512 tokens, batch 8. It downloads the weights from Hugging Face (network); the code pins no revision. No paid API. It is CPU-heavy and no chain script runs it. It removes only the seven ML section tags (`[SUMMARY]`, `[MAIN]`, `[STRENGTHS]`, `[WEAKNESSES]`, `[QUESTIONS]`, `[LIMITATIONS]`, `[OTHER]`). It appends to `--out` and skips the ids it holds.

## Notes

- The ICML section tags, for example `[CLAIMS AND EVIDENCE]` and `[THEORY]`, stay in the text that the model reads.
  All 11,500 input rows have such a tag.
- The line order is the scoring order, not the input order.
- Git LFS stores it.

## Read by

`src/validate.py rq1_evidence`; `src/figures.py by_year` (list `COCODET`), which `figures.py table` calls.
