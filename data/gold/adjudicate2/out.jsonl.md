# data/gold/adjudicate2/out.jsonl

Reference grouping of the reference claims. 60 lines, one per panel of `in.jsonl`. Fields: `panel_id`, `clusters`
(each cluster: `problem`, `claim_ids`). Every claim id of the panel occurs once. Mean groups per panel: 23.8 for
same-paper panels, 32.2 for cross-paper panels.

## Make

```
uv run python -I src/gold.py judge_groups --out <dir>
```

Step 7 of `scripts/gold.sh <dir>`. It sends one call per panel to `settings.GOLD_JUDGE_MODEL` (`opus-5.5`) through
OpenRouter, with the rules of `INSTRUCTIONS.md` and a strict JSON schema. It retries a panel up to 3 times when the
groups do not cover each claim id once. It writes rows in input order and retries missing panels on a rerun. Paid
API; needs network and the OpenRouter key. `scripts/gold.sh` does not write into `data/gold`.

## Notes

- Model output (Opus 5.5), not human labels. A rerun gives different groups.
- Rows follow the order of `in.jsonl`.
- Git LFS stores it.

## Read by

`src/gold.py judge_groups` (skips panels that it holds). The paper appendix (`app:reference`) describes it.

`src/validate.py reference` (reference overlap per panel).
