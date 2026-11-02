# data/other/plos_papers.parquet

PLOS journals (ONE, Biology, Computational Biology, Genetics, Medicine, Pathogens, NTD, Global Public Health,
Digital Health): one row per paper in `plos_reviews.parquet`. 3,756 papers.

Columns: `conf`, `field`, `journal`, `paper_id`, `title`, `licence`, `n_split`, `year`, `n_reviews`. `n_reviews`
counts the paper's rows in `plos_reviews.parquet`. `licence` is the article licence. `n_split` is the number of
reviewer texts that the parser found in the decision letter.

## Make

`src/fetch.py plos` writes it in the same run as `plos_reviews.parquet`. See `plos_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which counts paper licences for `reports/other_inventory.txt`.
