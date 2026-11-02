# data/other/f1000_papers.parquet

F1000Research, Wellcome Open Research and Open Research Europe: one row per paper kept in `f1000_reviews.parquet`. 4,400 papers.

Columns: `conf`, `field`, `journal`, `paper_id`, `title`, `licence`, `year`, `n_reviews`. `n_reviews` counts this paper's rows in `f1000_reviews.parquet`; `licence` is the article
licence.

The same run as `f1000_reviews.parquet` writes this file; see `f1000_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_papers.parquet`.
