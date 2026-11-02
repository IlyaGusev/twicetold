# data/gold/adjudicate/out_00.jsonl … out_11.jsonl

Reference claims: the judge's claim list for each review of `in_XX.jsonl`, same order. One line per review:
`review_id`, `claims` (each claim: `claim`, `category`, `target`, `specific`, `support`; `support` counts the council
members that list the claim).

## Make

No command in the repo makes these files. They are saved labels that Claude Opus 5.5 agents wrote by the rules of
`INSTRUCTIONS.md`, one agent per input file.

```
uv run python -I src/gold.py judge_claims --out <dir>
```

This step of `scripts/gold.sh` writes files of the same form. It calls the judge model (`settings.GOLD_JUDGE_MODEL`,
`opus-5.5`) through the paid OpenRouter API, needs `OPENROUTER_API_KEY`, and uses `INSTRUCTIONS.md` as the prompt with
a strict JSON schema and the 25-word check. Its labels differ from these files.

## Notes

- LLM labels, not human labels.
- Git LFS stores them.

## Read by

`src/gold.py merge_claims`; `tests/test_gold.py` (copies them as the input of `merge_claims`).
