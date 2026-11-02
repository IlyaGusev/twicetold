# data/gold/adjudicate2/in.jsonl

Input of the grouping judgment. 60 lines, one per panel, in the order of `panels_tune.jsonl`. Fields: `panel_id`,
`kind`, `year`, `n_reviews`, `claims` (reference claims with ids `R<review>-<claim>`, category and specific flag),
`council` (the `problem` and `claim_ids` of each cluster of `sol-or`, `opus-5.5` and `kimi-k3.1-pro`).
`INSTRUCTIONS.md` in this folder holds the rules.

59 panels have all 3 council entries. Panel `B34` has no `opus-5.5` entry, because `clusters_council.jsonl` holds an
error row for it.

## Make

```
uv run python -I src/gold.py group_inputs --out <dir>
```

Step 6 of `scripts/gold.sh <dir>`. It reads `panels_tune.jsonl`, `adjudicate/out.jsonl` and `clusters_council.jsonl`
in `<dir>`, and also writes `adjudicate2/INSTRUCTIONS.md` from `src/prompts/gold_groups.j2`. Deterministic, no
network, no paid API. `scripts/gold.sh` does not write into `data/gold`. `tests/test_gold.py` rebuilds the file from
the committed inputs and checks that it is byte-identical.

## Notes

- Git LFS stores it.

## Read by

`src/gold.py judge_groups` (input of the judge); `tests/test_gold.py`.
