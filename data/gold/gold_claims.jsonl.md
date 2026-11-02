# data/gold/gold_claims.jsonl

All reference claims: `adjudicate/out_00.jsonl` … `out_11.jsonl` joined in file order. 240 reviews, 1,810 claims
(2 reviews have none). `support`: 3 for 1,749 claims, 2 for 61. 86.5% of claims are `specific`.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

`src/gold.py merge_claims`. Original run: 2026-10-31 15:11 UTC.

## Check

`tests/test_gold.py` rebuilds this file from the inputs above. The result is byte-identical.
