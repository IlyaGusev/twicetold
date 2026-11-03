# data/sample_reviews.parquet

The main ICLR and NeurIPS review sample and the pass-1 input of the main chain. One row per review: 116,847 reviews
of 30,937 papers.

| Column | Meaning |
|---|---|
| `conf`, `year` | Venue (`ICLR`, `NeurIPS`) and conference year |
| `paper_id`, `review_id` | OpenReview forum id and note id |
| `text` | Review text, with the section tags of `reviews.parquet` |

Rows are sorted by `conf`, `year`, `paper_id`, `review_id`. Each paper has 3 to 5 reviews.

| Year | ICLR papers | ICLR reviews | NeurIPS papers | NeurIPS reviews |
|---|---|---|---|---|
| 2017 | 480 | 1,735 | – | – |
| 2018 | 918 | 2,771 | – | – |
| 2019 | 1,412 | 4,493 | – | – |
| 2020 | 2,127 | 6,546 | – | – |
| 2021 | 2,594 | 10,019 | 1,000 | 3,852 |
| 2022 | 2,617 | 10,188 | 1,000 | 3,662 |
| 2023 | 3,789 | 14,271 | 1,000 | 4,412 |
| 2024 | 3,000 | 11,596 | 1,000 | 3,895 |
| 2025 | 3,000 | 12,046 | 1,000 | 4,031 |
| 2026 | 6,000 | 23,330 | – | – |

## Make

No command in the repo rebuilds this file. Treat it as the fixed sample. `uv run python -I src/panels.py ml` writes a
file with this name, but on the current `reviews.parquet` it draws a different sample.

The sample rules (`ml` in `src/panels.py`, seed 2026):

- Eligible papers: not withdrawn, 3 or more reviews, decision not `desk_reject`.
- Papers per cell: all eligible papers, except ICLR 2024 and 2025 (3,000 each), ICLR 2026 (6,000) and NeurIPS
  2021–2025 (1,000 each), drawn at random without replacement.
- At most 5 reviews per paper, drawn at random.

## Notes

414 reviews are not in the current `reviews.parquet`: ICLR 2017 (231), ICLR 2019 (180), ICLR 2020 (3). Its duplicate
(paper, text) filter removes them. The other 116,433 reviews are there, with the same text, paper, venue and year.

## Read by

`scripts/extract_all.sh` (pass-1 input of the main chain), `src/panels.py` (`robust`), `src/between.py`
(`embed_reviews`, `load_reviews`), `src/meld.py` (`input`), `src/gold.py` (`sample`).
