# data/meld_scores.jsonl

MELD scores for the 187,610 texts of `meld_input.parquet`. One JSON object per line: `key` (string, as in
`meld_input.parquet`), `score` (float, MELD detection score; null for one empty text), `n_tokens` (int), `family`
(predicted generator family), `task` (predicted operation: `generate`, `polish_formalize`, `paraphrase_rewrite`,
`expand`, `summarize`, `humanize_mimic`).

## Make

```
uv run python -I src/meld.py scores
```

It downloads MELD (`settings.MELD_REPO` at `settings.MELD_REVISION`) from Hugging Face (network) and runs it in fp32
on a CUDA GPU. It removes the section tags of `src/ai.py` (`TAGS`), cuts each text into 2,048-token windows, batches
windows across texts under a 16,384-token budget, and pools token scores with MELD's top-quantile mean (rho 0.25).
`src/meld.py check` compares the batched scores with MELD's own scorer on 64 texts (max difference 1.5e-05). It took
about 4.5 hours on one GPU. No paid API. No chain script runs it. It appends to the output and skips the keys it holds.

## Notes

- The line order is the scoring order (shortest text first), not the input order.
- Git LFS stores it.

## Read by

`src/meld.py flags`.
