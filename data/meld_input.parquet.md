# data/meld_input.parquet

The texts that MELD scores. One row per text: 195,510 rows, unique `key`. Columns: `key` (`<set>:<review_id>`),
`set`, `review_id`, `conf`, `year`, `text`.

| `set` | Rows | Source |
|---|---|---|
| `sample` | 116,847 | All of `sample_reviews.parquet` (ICLR, NeurIPS) |
| `icml` | 11,500 | All of `icml_sample.parquet` |
| `journal` | 41,069 | All of `other_reviews.parquet` |
| `intel_ai` | 5,000 | LLM reviews (`label == 1`) of `intel_reviews.parquet`, 1,000 per LLM, `random_state=0`; `conf` is ICLR |
| `rewrite` | 13,194 | Journal rows of `ai_reference_gpt-6-luna.jsonl` (no ICLR or NeurIPS rows) |
| `polish` | 2,100 | All of `ai_polish_gpt-6-luna.jsonl` |
| `rewrite_panel` | 1,600 | All of `rewrite_test/rewrites_gpt-6-luna.jsonl` |
| `polish_panel` | 1,600 | All of `rewrite_test/polishes_gpt-6-luna.jsonl` |
| `rewrite_panel_second` | 1,200 | All of `rewrite_test/rewrites_kimi-k3.1-pro.jsonl` |
| `polish_panel_second` | 1,200 | All of `rewrite_test/polishes_kimi-k3.1-pro.jsonl` |
| `paper_review` | 100 | All of `paper_reviews/reviews_gpt-6-luna.jsonl` |
| `paper_review_second` | 100 | All of `paper_reviews/reviews_kimi-k3.1-pro.jsonl` |

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
