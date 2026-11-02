# data/panels_icml.jsonl

ICML 2025 and 2026 panels: one panel per sampled paper. 3,000 panels with 11,500 reviews. Keys: `panel_id`
(`icml_<paper_id>`) and `review_ids`. Sorted by paper id.

Panel sizes: 3 reviews (576), 4 (2,348), 5 (76). All review ids are in `icml_sample.parquet`.

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/panels.py icml
```

It reads `icml2025_reviews.parquet` and `icml2026_reviews.parquet` and writes `icml_sample.parquet` and this file
(seed 11). For each year: keep reviews with more than 200 characters; drop duplicate (paper, text) pairs; pick 1,500
papers with 3 or more reviews at random; shuffle their reviews (`random_state=0`); keep at most 5 reviews per paper.
`by_paper_panels` groups the sample by paper. It is deterministic and needs no network.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_icml`), `src/outcomes.py` (`SOURCE_PANEL_FILES`).
