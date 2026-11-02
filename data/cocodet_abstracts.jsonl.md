# data/cocodet_abstracts.jsonl

CoCoDet class probabilities for the title and abstract of 29,518 ICLR and NeurIPS papers
(`cocodet_input_abstracts.parquet`). One JSON object per line: `review_id` (the paper id, string), `human`, `mix`,
`ai` (floats that sum to 1). The line order is the scoring order, not the input order.

## How we got it

1. Model: the same as `cocodet_main.jsonl` (CoCoDet `khaaaaaan/CoCoDet` `CoCoDet.bin`, cached revision
   `7f60a5dcdad3af02767cb57da842ead41424c89d`; base `answerdotai/ModernBERT-base`).
2. Code: `src/ai.py cocodet` as of 2026-10-30 (the working tree after commit `c57b0a8`): 512 tokens, batch 8,
   2 CPU threads, this machine.
3. Command, Claude Code session `881519d6` (project `~/ideas`), 2026-10-30 22:08:16 UTC:

   ```
   OMP_NUM_THREADS=2 nice -n 19 uv run python -I src/ai.py cocodet data/cocodet_input_abstracts.parquet --name abstracts
   ```

   The log went to the session scratchpad,
   `/tmp/claude-1000/-home-yallen-ideas/881519d6-0823-4954-a4c3-bf0b4f7ffed9/scratchpad/cocodet_abstracts.log`
   (not permanent): "0 done, 29518 to score", 3,690 of 3,690 batches in 3:44:59. The last write was on 2026-10-31 at
   01:53:24 UTC. No paid API.

## Reproduce

```
DATA_DIR=<dir> uv run python -I src/ai.py cocodet data/cocodet_input_abstracts.parquet --name abstracts
```

## Check

On 2026-11-02 we did not rerun the model. We checked the file against `cocodet_input_abstracts.parquet`:

- 29,518 lines, 29,518 distinct ids; the id set equals the input id set.
- No NaN. Each probability is in [6.3e-14, 1.0]; each row sums to 1.000000.
- Largest class: human 25,449; mix 3,525; ai 544.

## Read by

`src/validate.py paper_side`: score = `ai + mix` per paper.
