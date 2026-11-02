# data/gold/adjudicate2/out_00.jsonl … out_05.jsonl

Reference grouping of the gold claims. Each file holds the judge's grouping for the panels of `in_XX.jsonl` with the
same number, in the same order. One line per panel: `panel_id`, `clusters` (each cluster: `problem`, `claim_ids`).
Every claim id of the panel occurs once. 60 panels in 6 files. Mean groups per panel: 23.8 for same-paper panels,
32.2 for cross-paper panels.

## Make

```
uv run python -I src/gold.py judge_groups --out <dir>
```

Step 8 of `scripts/gold.sh <dir>`. It sends one call per panel to `settings.GOLD_JUDGE_MODEL` (`opus-5.5`) through
OpenRouter, with the rules of `src/prompts/gold_groups.j2` and a strict JSON schema. It retries a panel up to 3
times when the groups do not cover each claim id once. Paid API; needs network and the OpenRouter key. It skips an
`out_XX.jsonl` that exists. `scripts/gold.sh` does not write into `data/gold/`.

## Notes

- Model output (Opus 5.5), not human labels. A rerun gives different groups.

## Read by

Nothing reads it. The paper appendix (`app:reference`) describes it.
