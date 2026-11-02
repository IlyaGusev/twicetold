# data/other/scipost_papers.parquet

SciPost (physics and related fields): one row per paper in `scipost_reviews.parquet`. 3,052 papers.

Columns: `conf`, `field`, `journal`, `paper_id`, `title`, `licence`, `year`, `n_reviews`. `n_reviews` counts the
paper's rows in `scipost_reviews.parquet`. `licence` is CC BY 4.0 for every paper.

## Make

`src/fetch.py scipost` writes it in the same run as `scipost_reviews.parquet`. See `scipost_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_papers.parquet`.
