# data/gold/adjudicate2/in_00.jsonl … in_05.jsonl

Inputs of the grouping judgment: 6 files of 10 panels; file `i` holds panels `i`, `i+6`, `i+12` … of
`panels_tune.jsonl`. One line per panel: `panel_id`, `kind`, `year`, `n_reviews`, `claims` (reference claims with ids
`R<review>-<claim>`, category and specific flag), `council` (the `problem` and `claim_ids` of each cluster of
`sol-or`, `opus-5.5` and `kimi-k3.1-pro`). `INSTRUCTIONS.md` in this folder holds the rules.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

`src/gold.py group_inputs`. Original run: 2026-10-31 15:26 UTC.

## Check

`tests/test_gold.py` rebuilds this file from the inputs above. The result is byte-identical.

## Read by

The grouping judges (`out_XX.jsonl.md`).
