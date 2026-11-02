# data/gold/blind_key.json

The 40 same-paper panels of the reference set: code (`B00`–`B39`) → `paper_id`, `year`, `review_ids` (4 reviews
each). 10 ICLR papers in each of 2018, 2022, 2024 and 2026.

## How we got it

`validate.py blind_set` drew it on 2026-10-29 from `sample_reviews.parquet` for an earlier blind annotation study and
saved it as `data/blind/blind_key.json`. Commit `f948ee5` (2026-11-01) deleted `data/blind/`. On 2026-11-02 we
restored the file unchanged with `git show f948ee5^:data/blind/blind_key.json`, because `src/gold.py sample` needs it.

## Read by

`src/gold.py sample`.
