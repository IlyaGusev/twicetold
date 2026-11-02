# data/gold/claims_gold_tune.jsonl

`gold_claims.jsonl` in the pipeline's claim format: `review_id`, `model` (`gold`), `claims` without `support`. This
lets `src/extract.py pass2` group the reference claims.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

`src/gold.py merge_claims`. Original run: 2026-10-31 15:19 UTC.

## Check

`tests/test_gold.py` rebuilds this file from the inputs above. The result is byte-identical.

## Read by

`src/extract.py pass2 --claims_files claims_gold_tune.jsonl`, `src/gold.py group_inputs`.
