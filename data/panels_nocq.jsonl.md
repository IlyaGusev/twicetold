# data/panels_nocq.jsonl

ICLR 2023 panels for the reviews without the "clarity, quality, novelty and reproducibility" field. 3,789 panels with
14,271 reviews: the ICLR 2023 panels of `panels.jsonl`, with the same reviews in the same order. Keys: `panel_id`
(`nocq_<paper_id>`) and `review_ids` (`nocq_<OpenReview note id>`). The texts are in `nocq_reviews.parquet`.

## Make

`scripts/inputs.sh` makes it. The step that writes it:

```
uv run python -I src/panels.py nocq
```

It reads `sample_reviews.parquet` and `panels.jsonl` (`--data` changes the folder). One run writes
`panels_nocq.jsonl` and `nocq_reviews.parquet`. It is deterministic and needs no network.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_nocq`). `src/outcomes.py panels` scores the panels into
`data/overlap/panel_outcomes_gpt-6-luna-high_nocq.parquet`; `src/validate.py forms` compares them with the main 2023
panels of the same papers.
