# data/abstract_emb.npy

OpenAI `text-embedding-3-small` vectors of paper titles and abstracts. Shape (44,406, 1,536), float32. Row `i` is the
paper in row `i` of `paper_areas.parquet` (its column `row` equals `i`).

The text of each row is `title + "\n" + abstract`, cut to 6,000 characters. The papers are the panels of
`panels.jsonl` plus ICLR 2026 papers with a `primary_area`, in `papers.parquet` order. Papers with 50 characters of
text or fewer are not included.

## Make

```
uv run python -I src/areas.py classify --out <dir>
```

This writes `abstract_emb.npy` and `paper_areas.parquet` into `<dir>`. It calls the paid OpenAI embeddings API
(`settings.EMBED_MODEL`) and needs `OPENAI_API_KEY`. No chain script runs it.

## Notes

- With the current `panels.jsonl` the command embeds 44,399 papers, not 44,406. Thus a rerun does not give this file.
- Git LFS stores it.

## Read by

Nothing reads it. Column `row` of `paper_areas.parquet` points into it.
