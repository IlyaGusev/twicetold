# data/other/plos_papers.parquet

PLOS journals (ONE, Biology, Computational Biology, Genetics, Medicine, Pathogens, NTD, Global Public Health, Digital Health): one row per paper kept in `plos_reviews.parquet`. 3,756 papers.

Columns: `conf`, `field`, `journal`, `paper_id`, `title`, `licence`, `n_split`, `year`, `n_reviews`. `n_reviews` counts this paper's rows in `plos_reviews.parquet`; `licence` is the article
licence.

The same run as `plos_reviews.parquet` writes this file; see `plos_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which merges all five sources into `data/other_papers.parquet`.
