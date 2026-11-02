# data/review_emb.npy

OpenAI `text-embedding-3-small` vectors of review texts. Shape (116,847, 1,536), float32. Row `i` is row `i` of
`sample_reviews.parquet`.

The text of each row is the `text` column (with section tags), cut to 12,000 characters. When the API rejected a
batch of 64, the code cut every text in that batch to half its length and tried again. The run did not log which
batches it cut.

## How we got it

1. Script: `src/analysis_embed.py embed` (removed in commit `c57b0a8`; `src/between.py embed --reembed` now does the
   same steps, with `REVIEW_EMBED_CHARS = 12000`).
2. Command, Claude Code session `1e2ee420` (project `~/twicetold`), 2026-10-29 08:06:13 UTC:

   ```
   set -a && . ./.env && set +a && (THREADS=4 uv run python -I src/analysis_embed.py embed > logs/embed_reviews.log 2>&1 && uv run python -I src/analysis_embed.py analyze > reports/embed_gpt-6-luna.txt 2>&1; echo EMBED DONE >> logs/embed_reviews.log)
   ```

   Three earlier runs failed and wrote no file: 07:50 UTC (rate limit), 07:51 UTC (cut at 24,000 characters; a text
   over 8,192 tokens) and 07:57 UTC (a batch failed after 8 tries). Model:
   `text-embedding-3-small` through the OpenAI API (`settings.EMBED_MODEL`). The log `logs/embed_reviews.log` says
   "embedded 116847". The last write was on 2026-10-29 at 08:13:55 UTC. The session did not log the cost.

## Reproduce

```
uv run python -I src/between.py embed --reembed --data <dir>
```

`<dir>` must hold a copy of `sample_reviews.parquet` and the other files that the analysis step of `between.py embed`
reads. This calls the paid OpenAI API.

## Check

On 2026-11-02:

- Shape (116,847, 1,536), float32, no NaN, no zero rows, row norms 0.9993–1.0007.
- The current `sample_reviews.parquet` has 116,847 rows.
- We embedded rows 0, 58,423 and 116,846 of the current `sample_reviews.parquet` again (1,163 tokens, about
  $0.00002). The dot products with the stored rows are 1.0007, 1.0003 and 1.0001; with a wrong row, 0.68. The row
  order still matches the current table.

## Read by

`src/between.py embed` (`load_reviews`).
