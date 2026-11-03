# data/meld_input.parquet

The texts that MELD scores. One row per text: 187,610 rows, unique `key`. Columns: `key` (`<set>:<review_id>`),
`set`, `review_id`, `conf`, `year`, `text`.

| `set` | Rows | Source |
|---|---|---|
| `sample` | 116,847 | All of `sample_reviews.parquet` (ICLR, NeurIPS) |
| `icml` | 11,500 | All of `icml_sample.parquet` |
| `journal` | 41,069 | All of `other_reviews.parquet` |
| `intel_ai` | 5,000 | LLM reviews (`label == 1`) of `intel_reviews.parquet`, 1,000 per LLM, `random_state=0`; `conf` is ICLR |
| `rewrite` | 13,194 | Journal rows of `ai_reference_gpt-6-luna.jsonl` (no ICLR or NeurIPS rows) |

## Make

`scripts/inputs.sh` runs this step:

```
uv run python -I src/meld.py input
```

Deterministic, no network, no paid API.

## Notes

- Not in git (`.gitignore`). `scripts/inputs.sh` rebuilds it.
- The text keeps its section tags; `src/meld.py scores` removes them.

## Read by

`src/meld.py` (`check`, `scores`, `flags`).
