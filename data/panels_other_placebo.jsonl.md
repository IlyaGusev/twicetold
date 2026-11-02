# data/panels_other_placebo.jsonl

Same-journal cross-paper control panels for the journal sources. Each panel holds 3 reviews of 3 different papers of
one journal, source and year. 6,450 panels (150 for each of 43 source-year cells) with 19,350 review ids (14,783
unique). Keys: `panel_id` (`otherplacebo_<conf>_<year>_<k>`) and `review_ids`.

Panels per source: BMC-Med 1,200; Copernicus 1,350; F1000 1,350; PLOS 1,200; SciPost 1,350. All review ids are in
`other_reviews.parquet`.

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/panels.py other
```

It reads `other_reviews.parquet` and keeps at most 5 reviews per paper, drawn at random (`cap_reviews_per_paper`,
seed 2027). Inner function `placebo`: in each (source, year) cell, take the journals with 3 or more papers, except the
Copernicus preprint server `EGUsphere`. 150 times: pick a journal with weight equal to its paper count, pick 3 of its
papers, then 1 review of each paper. The same run writes `other_pass1.parquet`, `panels_other.jsonl` and
`reports/other_placebo_preprints.csv` (preprint share per cell). It is deterministic and needs no network.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_other`), `src/outcomes.py` (`SOURCE_PANEL_FILES`; the
`otherplacebo_` prefix marks a control panel).
