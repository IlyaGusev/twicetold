# data/review_emb.npy

OpenAI `text-embedding-3-small` vectors of review texts. Shape (116,847, 1,536), float32. Row `i` is row `i` of
`sample_reviews.parquet`.

The text of each row is the `text` column (with section tags), cut to 12,000 characters (`REVIEW_EMBED_CHARS`).

## Make

```
uv run python -I src/between.py embed --reembed
```

`embed_reviews` reads `sample_reviews.parquet` and overwrites `data/review_emb.npy` (`--data` changes the folder).
Then `embed` runs its analysis and writes `reports/overlap/embed_<model>.csv`. It calls the paid OpenAI API and needs
`OPENAI_API_KEY` (`settings.EMBED_MODEL`).

## Notes

When the API rejects a batch of 64, the code cuts every text in that batch to half its length and tries again. The
file does not record which rows have a cut text.

## Read by

`src/between.py` (`load_reviews`, used by `embed`).
