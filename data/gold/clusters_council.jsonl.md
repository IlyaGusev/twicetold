# data/gold/clusters_council.jsonl

Council grouping of the reference claims (`adjudicate/out.jsonl`) for the 60 panels of `panels_tune.jsonl`, from
each model of `settings.GOLD_COUNCIL`: `sol-or` (OpenRouter `openai/gpt-6.1-sol`), `opus-5.5`
(`anthropic/claude-opus-5.5`) and `kimi-k3.1-pro` (`moonshotai/kimi-k3.1-pro`). 180 lines: one line per model and
panel, key `model` + `panel_id`. Fields: `panel_id`, `model`, `review_ids`, `targets`, `clusters` (each cluster:
`target_id`, `problem`, `claim_ids` such as `R2-5`).

179 lines hold a grouping. 1 line is an error row: `opus-5.5` for panel `B34`. `B34` has no Opus grouping, so its
grouping-judge input has no Opus entry.

## Make

```
uv run python -I src/gold.py council_groups --out <dir>
```

Step 5 of `scripts/gold.sh <dir>`. It runs `src/extract.py pass2` on `panels_tune.jsonl` with
`--claims_file adjudicate/out.jsonl` and tag `_tune_gold` for each council model, two times, so the second run retries
failed panels. Then it joins the part files `clusters_<model>_tune_gold.jsonl` into this file
(`extract.merge_rows`) and deletes them. Reasoning effort medium. Paid API; needs network and the OpenRouter key.

## Notes

- Model output; a rerun gives different groups.
- Rows have no `effort` field. Rows from the current code have one.
- Git LFS stores it.

## Read by

`src/gold.py group_inputs`.
