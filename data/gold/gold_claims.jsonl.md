# data/gold/gold_claims.jsonl

All reference claims: `adjudicate/out_00.jsonl` … `out_11.jsonl` joined in file order. One line per review:
`review_id`, `claims` (each claim: `claim`, `category`, `target`, `specific`, `support`). 240 reviews, 1,810 claims;
2 reviews have none. `support` is 3 for 1,749 claims and 2 for 61. 86.5% of claims are `specific`.

## Make

```
uv run python -I src/gold.py merge_claims --out <dir>
```

Step 5 of `scripts/gold.sh <dir>`. The same step writes `claims_gold_tune.jsonl`. Deterministic, no API.
`tests/test_gold.py` rebuilds the file and checks that it is byte-identical.

## Read by

Nothing in `src/` reads it. `tests/test_gold.py` compares it.
