# data/match_gpt-6-luna-high/clusters_opus-5.5.jsonl

Claim match between `gpt-6-luna-high` and the council reference. Claude Opus 5.5 (`opus-5.5`, reasoning effort
medium) groups the reference and Luna claims of one review with the pass-2 prompt (`src/prompts/pass2.j2`), as if they
came from two reviews of one paper. A claim is matched if its cluster holds a claim of the other list. Fields as in
`data/clusters_gpt-6-luna-high.jsonl`: `panel_id` (= review id), `model`, `effort`, `review_ids`, `targets`,
`clusters`.

246 lines: 238 reviews with a grouping and 8 error rows. Reviews `bhOZlJVyWY5` and `GE9GtXRABU` failed in all 6
attempts (truncated or empty JSON), so they have no grouping.

## Make

```
uv run python -I src/gold.py match_claims --model gpt-6-luna-high --budget 15
```

Paid API (OpenRouter); a rerun retries the failed reviews. Cost of all runs: $5.0.

## Notes

- Model output; a rerun gives different groups.
- Opus 5.5 also judged the reference labels, so the matcher is not independent of them.
- Git LFS stores it.

## Read by

`src/validate.py reference` (recall and precision per panel), through `metrics.load_jsonl` (skips error rows).
