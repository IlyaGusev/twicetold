# data/gold/adjudicate/in.jsonl

Input of the claim judgment. 240 lines, one per review, in panel order of `panels_tune.jsonl` (the 4 reviews of each
panel in turn). Fields: `review_id`, `text`, `council` (the claim lists of `sol-or`, `opus-5.5` and `kimi-k3.1-pro`
from `claims_council.jsonl`). `INSTRUCTIONS.md` in this folder holds the judgment rules.

## Make

```
uv run python -I src/gold.py claim_inputs --out <dir>
```

Step 3 of `scripts/gold.sh <dir>`. It reads `panels_tune.jsonl`, `tune_reviews.parquet` and `claims_council.jsonl` in
`<dir>`, and also writes `adjudicate/INSTRUCTIONS.md` from `src/prompts/gold_claims.j2`. Deterministic, no network,
no paid API. `scripts/gold.sh` does not write into `data/gold`. `tests/test_gold.py` rebuilds the file from the
committed inputs and checks that it is byte-identical.

## Notes

- Git LFS stores it.

## Read by

`src/gold.py judge_claims` (input of the judge); `tests/test_gold.py`.
