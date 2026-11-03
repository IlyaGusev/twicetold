# data/ai_reference_gpt-6-luna.jsonl

LLM rewrites of human reviews, the `rewrite` recall set of the MELD check. For each sampled human review from 2018–2021, `gpt-6-luna`
wrote an outline of its points, then a new review from the outline only. One line per review: 19,194 rows.

| Field | Meaning |
|---|---|
| `review_id` | Id of the human review in `reviews.parquet` or `other_reviews.parquet` |
| `conf`, `year` | Source and year of the human review |
| `outline` | Bullet list from step 1, one `- ` line per point |
| `text` | AI-written review from step 2 |

Rows have no `model` or `effort` field.

| Source | Rows | Years |
|---|---|---|
| BMC-Med | 2,636 | 2018–2021 |
| Copernicus | 2,634 | 2018–2021 |
| F1000 | 3,000 | 2018–2021 |
| ICLR | 3,000 | 2018–2021 |
| NeurIPS | 3,000 | 2021 |
| PLOS | 3,000 | 2019–2021 |
| SciPost | 1,924 | 2018–2021 |

## Make

```
uv run python -I src/ai.py reference --out <new file>
```

It samples up to 3,000 reviews per source (`reference_sample`, `random_state=0`), cuts each at 14,000 characters,
and makes two calls per review with the prompts `src/prompts/ai_outline.j2` and `ai_write.j2`. It calls the paid
OpenAI API with `settings.AI_REFERENCE_MODEL` (`gpt-6-luna`) at effort low
(`settings.REASONING_EFFORTS["gpt-6-luna"].pass1`) and needs `OPENAI_API_KEY`. Model output is not deterministic. No
chain script runs it. Without `--out` it appends to this file and skips the ids it holds.

## Notes

- The current review tables give a different sample, so a rerun does not give the same rows.
- 598 rows (PLOS 565, Copernicus 23, SciPost 10) have a `review_id` that is not in the current `reviews.parquet` or
  `other_reviews.parquet`.
- Git LFS stores it.

## Read by

`src/meld.py input`, which keeps the journal rows (not ICLR or NeurIPS) as the `rewrite` set.
