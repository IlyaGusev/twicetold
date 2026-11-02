# data/gold/adjudicate/in_00.jsonl … in_11.jsonl

Inputs of the claim judgment: 12 files of 20 reviews, 240 in total, in panel order (the 4 reviews of a panel stay in
one file). One line per review: `review_id`, `text`, `council` (the claim lists of `sol-or`, `opus-5.5` and
`kimi-k3.1-pro` from `claims_<model>_tune.jsonl`). `INSTRUCTIONS.md` in this folder holds the judgment rules.

## Make

```
uv run python -I src/gold.py claim_inputs --out <dir>
```

This is the `claim_inputs` step of `scripts/gold.sh <dir>`. It reads `panels_tune.jsonl`, `tune_reviews.parquet` and
the council claim files in `<dir>`, and also writes `adjudicate/INSTRUCTIONS.md` from `src/prompts/gold_claims.j2`.
Deterministic, no network, no paid API. `scripts/gold.sh` does not write into `data/gold`. `tests/test_gold.py`
rebuilds these files from the committed inputs and checks that they are byte-identical.

## Read by

`src/gold.py judge_claims` (input of the judge) and `merge_claims` (counts the files); `tests/test_gold.py`.
