# data/other/bmc_papers.parquet

BMC Medicine: one row per paper kept in `bmc_reviews.parquet`. 2,612 papers.

Columns: `conf`, `field`, `journal`, `paper_id`, `title`, `licence`, `year`, `n_reviews`. `n_reviews` counts this paper's rows in `bmc_reviews.parquet`; `licence` is the article
licence.

The same run as `bmc_reviews.parquet` writes this file; see `bmc_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_papers.parquet`.
