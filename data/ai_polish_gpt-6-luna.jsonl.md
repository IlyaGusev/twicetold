# data/ai_polish_gpt-6-luna.jsonl

LLM-polished human reviews, the `polish` set of the MELD label check. For each sampled human review from 2018–2021,
`gpt-6-luna` polished the language and kept every point (prompt `src/prompts/ai_polish.j2`). One line per review:
2,100 rows, 300 per source. Fields: `review_id`, `conf`, `year`, `text` (the polished review).

## Make

```
uv run python -I src/ai.py polish
```

`scripts/extract_all.sh` runs it after the rewrite chain. It takes the first 300 rows per source of `reference_sample`
(`random_state=0`), cuts each text at 14,000 characters and makes one call at effort low. Paid OpenAI API
(`OPENAI_API_KEY`); model output is not deterministic. It appends and skips the ids it holds.

## Notes

- Git LFS stores it.

## Read by

`src/meld.py input` (set `polish`).
