# data/gold/claims_gold_tune.jsonl

`gold_claims.jsonl` in the pipeline claim format. One line per review: `review_id`, `model` (`gold`), `claims`
(each claim: `claim`, `category`, `target`, `specific`; no `support`). 240 lines. With this format,
`src/extract.py pass2` can group the reference claims.

## Make

```
uv run python -I src/gold.py merge_claims --out <dir>
```

Step 5 of `scripts/gold.sh <dir>`. The same step writes `gold_claims.jsonl`. Deterministic, no API.
`tests/test_gold.py` rebuilds the file and checks that it is byte-identical.

## Read by

`src/gold.py council_groups` (as `--claims_files` of `src/extract.py pass2`), `src/gold.py group_inputs`.
