# data/panels_other.jsonl

Journal panels: one panel per paper of `other_pass1.parquet`. 17,520 panels with 40,992 reviews. Keys: `panel_id`
(`other_<paper_id>`) and `review_ids`.

Panel sizes: 2 reviews (12,637), 3 (3,984), 4 (729), 5 (170). All review ids are in `other_reviews.parquet`.

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/panels.py other
```

It reads `other_reviews.parquet`, keeps at most 5 reviews per paper, drawn at random (`cap_reviews_per_paper`, seed
2027), and groups them by paper (`by_paper_panels`). The same run writes `other_pass1.parquet`,
`panels_other_placebo.jsonl` and `reports/other_placebo_preprints.csv`. It is deterministic and needs no network.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_other`), `src/outcomes.py` (`SOURCE_PANEL_FILES`).
