# data/other/f1000_papers.parquet

F1000Research, Wellcome Open Research and Open Research Europe: one row per paper in `f1000_reviews.parquet`. 4,400
papers.

Columns: `conf`, `field`, `journal`, `paper_id`, `title`, `licence`, `year`, `n_reviews`. `n_reviews` counts the
paper's rows in `f1000_reviews.parquet`. `licence` is the article licence.

## Make

`src/fetch.py f1000` writes it in the same run as `f1000_reviews.parquet`. See `f1000_reviews.parquet.md`.

## Read by

`src/corpus.py other`, which counts paper licences for `reports/other_inventory.txt`.
