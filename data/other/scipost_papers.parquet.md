# data/other/scipost_papers.parquet

SciPost (physics and related fields): one row per paper kept in `scipost_reviews.parquet`. 3,052 papers.

Columns: `conf`, `field`, `journal`, `paper_id`, `title`, `licence`, `year`, `n_reviews`. `n_reviews` counts this paper's rows in `scipost_reviews.parquet`; `licence` is the article
licence.

The same run as `scipost_reviews.parquet` writes this file; see `scipost_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_papers.parquet`.
