# data/gold/adjudicate2/in_00.jsonl … in_05.jsonl

Inputs of the grouping judgment: 6 files of 10 panels; file `i` holds panels `i`, `i+6`, `i+12` … of
`panels_tune.jsonl`. One line per panel: `panel_id`, `kind`, `year`, `n_reviews`, `claims` (reference claims with ids
`R<review>-<claim>`, category and specific flag), `council` (the `problem` and `claim_ids` of each cluster of
`sol-or`, `opus-5.5` and `kimi-k3.1-pro`). `INSTRUCTIONS.md` in this folder holds the rules.

## Make

```
uv run python -I src/gold.py group_inputs --out <dir>
```

This is the `group_inputs` step of `scripts/gold.sh <dir>`. It reads `panels_tune.jsonl`, `claims_gold_tune.jsonl`
and `clusters_<model>_tune_gold.jsonl` in `<dir>`, and also writes `adjudicate2/INSTRUCTIONS.md` from
`src/prompts/gold_groups.j2`. Deterministic, no network, no paid API. `scripts/gold.sh` does not write into
`data/gold`. `tests/test_gold.py` rebuilds these files from the committed inputs and checks that they are
byte-identical.

## Read by

`src/gold.py judge_groups` (input of the judge); `tests/test_gold.py`.
