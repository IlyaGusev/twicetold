# data/cocodet_other.jsonl

CoCoDet class probabilities for the 10,042 journal reviews of `cocodet_input_other.parquet`. One JSON object per
line: `review_id` (string), `human`, `mix`, `ai` (floats that sum to 1). Largest class: human 9,553; mix 355;
ai 134.

## Make

```
uv run python -I src/cocodet.py scores --input data/cocodet_input_other.parquet --out <dir>/cocodet_other.jsonl
```

It runs CoCoDet (`settings.COCODET_REPO`, file `settings.COCODET_WEIGHTS_FILE`, base model `settings.COCODET_BASE_MODEL`) on CPU: 512 tokens, batch 8. It downloads the weights from Hugging Face (network); the code pins no revision. No paid API. It is CPU-heavy and no chain script runs it. It removes only the seven ML section tags (`[SUMMARY]`, `[MAIN]`, `[STRENGTHS]`, `[WEAKNESSES]`, `[QUESTIONS]`, `[LIMITATIONS]`, `[OTHER]`). It appends to `--out` and skips the ids it holds.

## Notes

- Journal tags, for example `[REPORT]` and `[REQUESTED CHANGES]`, stay in the text that the model reads. 1,990 input
  rows have such a tag.
- The line order is the scoring order, not the input order.
- Git LFS stores it.

## Read by

`src/figures.py by_year` (list `COCODET`), which `figures.py table` calls.
