# data/gold/clusters_opus-5.5_tune_gold.jsonl

Council grouping of the reference claims by `opus-5.5` (OpenRouter `anthropic/claude-opus-5.5`). One line per panel: `panel_id`,
`model`, `review_ids`, `targets`, `clusters` (each cluster: `target_id`, `problem`, `claim_ids` such as
`R2-5`). 61 lines: 59 grouped panels and 2 error rows, both for panel `B34`. `B34` has no Opus grouping, so its
grouping-judge input has no Opus entry.

## Make

```
uv run python -I src/gold.py council_groups --out <dir>
```

Step 6 of `scripts/gold.sh <dir>`. It runs `src/extract.py pass2` on `panels_tune.jsonl` with
`--claims_files claims_gold_tune.jsonl` and tag `_tune_gold` for each model of `settings.GOLD_COUNCIL`, two times, so
the second run retries failed panels. Reasoning effort medium. Paid API; needs network and the OpenRouter key.

## Notes

- Model output; a rerun gives different groups.

## Read by

`src/gold.py group_inputs`.
