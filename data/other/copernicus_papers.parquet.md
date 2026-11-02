# data/other/copernicus_papers.parquet

Copernicus (EGU open-discussion journals and EGUsphere): one row per paper kept in `copernicus_reviews.parquet`. 3,700 papers.

Columns: `conf`, `field`, `journal`, `paper_id`, `title`, `licence`, `year`, `n_reviews`. `n_reviews` counts this paper's rows in `copernicus_reviews.parquet`; `licence` is the article
licence.

The same run as `copernicus_reviews.parquet` writes this file; see `copernicus_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_papers.parquet`.
