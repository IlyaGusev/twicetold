# data/gold/adjudicate/in_00.jsonl … in_11.jsonl

Inputs of the claim judgment: 12 files of 20 reviews, 240 in total, in panel order (the 4 reviews of a panel stay in
one file). One line per review: `review_id`, `text`, `council` (the claim lists of `sol-or`, `opus-5.5` and
`kimi-k3.1-pro` from `claims_<model>_tune.jsonl`). `INSTRUCTIONS.md` in this folder holds the judgment rules.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

`src/gold.py claim_inputs`. Original run: 2026-10-31 15:05 UTC.

## Check

`tests/test_gold.py` rebuilds this file from the inputs above. The result is byte-identical.

## Read by

The claim judges (`out_XX.jsonl.md`).
