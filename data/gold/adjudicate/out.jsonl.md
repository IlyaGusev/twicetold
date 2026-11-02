# data/gold/adjudicate/out.jsonl

The reference claims: the judge's claim list for each review of `in.jsonl`, same order. 240 lines, one per review.
Fields: `review_id`, `claims` (each claim: `claim`, `category`, `target`, `specific`, `support`; `support` counts the
council members that list the claim).

1,810 claims; 2 reviews have none. `support` is 3 for 1,749 claims and 2 for 61. 86.5% of claims are `specific`.

## Make

No command in the repo makes this file. It holds saved labels that Claude Opus 5.5 agents wrote by the rules of
`INSTRUCTIONS.md`.

```
uv run python -I src/gold.py judge_claims --out <dir>
```

Step 4 of `scripts/gold.sh <dir>` writes a file of the same form. It calls the judge model
(`settings.GOLD_JUDGE_MODEL`, `opus-5.5`) through the paid OpenRouter API, needs the OpenRouter key, and uses
`INSTRUCTIONS.md` as the prompt with a strict JSON schema and the 25-word check. It writes rows in input order and
retries missing reviews on a rerun. Its labels differ from this file.

## Notes

- LLM labels, not human labels.
- Git LFS stores it.

## Read by

`src/gold.py council_groups` (as `--claims_file` of `src/extract.py pass2`), `group_inputs` and `judge_claims` (skips
reviews that it holds); `tests/test_gold.py` (copies it as an input of `group_inputs`).
