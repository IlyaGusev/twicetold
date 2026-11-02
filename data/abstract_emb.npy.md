# data/abstract_emb.npy

OpenAI `text-embedding-3-small` vectors of paper titles and abstracts. Shape (44,406, 1,536), float32. Row `i` is the
paper in row `i` of `paper_areas.parquet` (its column `row` equals `i`).

The text of each row is `title + "\n" + abstract`, cut to 6,000 characters. The papers are the panels of
`panels.jsonl` of 2026-10-28, plus ICLR 2026 papers with a `primary_area`, in `papers.parquet` order. Papers with
50 characters of text or fewer are left out.

## How we got it

1. Script: `src/areas.py` (removed in commit `c57b0a8`; restored on 2026-11-02 as `src/areas.py classify`, same logic). It embeds the
   texts in chunks of 256 with 2 threads, saves this file, then fits a logistic regression on the ICLR 2026 labels
   and writes `paper_areas.parquet`.
2. Command, Claude Code session `1e2ee420` (project `~/twicetold`), 2026-10-28 23:04:21 UTC:

   ```
   set -a && . ./.env && set +a && uv run python -I src/areas.py
   ```

   Model: `text-embedding-3-small` through the OpenAI API (`settings.EMBED_MODEL`). The last write was on
   2026-10-28 at 23:05:49 UTC. The session did not log the cost.

## Reproduce

```
uv run python -I src/areas.py classify --out <dir>
```

This calls the paid OpenAI API. With the current `panels.jsonl` it embeds 44,399 papers, not 44,406: 7 papers of
this file are no longer panels. The other 44,399 papers keep their relative order.

## Check

On 2026-11-02:

- Shape (44,406, 1,536), float32, no NaN, no zero rows, row norms 0.9994–1.0007.
- `paper_areas.parquet` has 44,406 rows and `row == 0..44,405`.
- We embedded rows 0, 22,203 and 44,405 again from the current `papers.parquet` (805 tokens, about $0.00002). The dot
  products with the stored rows are 0.9996, 1.0002 and 1.0000.

## Read by

No module loads it now. `paper_areas.parquet` (column `row`) points into it.
