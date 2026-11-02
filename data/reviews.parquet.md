# data/reviews.parquet

ICLR 2017–2026 and NeurIPS 2021–2025 official reviews from OpenReview. One row per review: 258,002 rows.

| Column | Meaning |
|---|---|
| `conf`, `year` | Venue (`ICLR`, `NeurIPS`) and conference year |
| `paper_id`, `review_id` | OpenReview forum id and note id |
| `reviewer` | OpenReview signature, for example `ICLR.cc/2024/Conference/Submission1/Reviewer_abcd` |
| `rating`, `confidence` | Leading integer of the rating and confidence fields |
| `text` | All free-text fields, joined in a fixed order with tags: `[SUMMARY]`, `[MAIN]`, `[STRENGTHS]`, `[WEAKNESSES]`, `[QUESTIONS]`, `[LIMITATIONS]`, `[OTHER]` |
| `weaknesses` | The weaknesses field alone, when the form has one |
| `form` | Names of the text fields that the review form had |
| `review_time`, `modified_time` | Creation and last edit time (`modified_time` only for ICLR 2026) |
| `withdrawn`, `decision`, `primary_area` | Copied from `papers.parquet` |

Rows per venue and year:

| Year | ICLR | NeurIPS |
|---|---|---|
| 2017 | 1,536 | – |
| 2018 | 2,779 | – |
| 2019 | 4,328 | – |
| 2020 | 6,715 | – |
| 2021 | 10,021 | 10,736 |
| 2022 | 10,202 | 10,330 |
| 2023 | 14,327 | 15,168 |
| 2024 | 22,098 | 16,643 |
| 2025 | 34,991 | 22,324 |
| 2026 | 75,804 | – |

## Make

```
uv run python -I src/corpus.py ml
```

It writes this file and `papers.parquet` into `data/` (`--out` changes the folder). No script in `scripts/` runs it.
It is deterministic and needs no network, but it needs these Hugging Face dumps in `RAW_ML_REVIEWS_DIR`:

| Local file | Source dataset |
|---|---|
| `arcade/train-*.parquet` | `ulab-ai/ResearchArcade-openreview-reviews` |
| `arcade/papers.parquet` | `ulab-ai/ResearchArcade-openreview-papers` |
| `iclr2026/heineman.parquet` | `davidheineman/iclr-2026` |
| `iclr2026/jerma_2026.parquet` | `JerMa88/ICLR_Peer_Reviews_2026` |

Steps (`from_arcade`, `from_iclr26`, `merge_ml` in `src/corpus.py`):

- ResearchArcade gives ICLR 2017–2025 and NeurIPS 2021–2025. It keeps rows whose title starts with
  `Official Review`, and drops EMNLP and ICLR before 2017.
- ICLR 2026 comes from the Heineman snapshot. The JerMa88 dump adds the questions field, creation and edit times,
  decisions, primary areas and final venues.
- It keeps reviews with more than 200 characters of text and drops duplicate review ids and duplicate
  (paper, text) pairs.
- It drops an older version of a review when the same reviewer has a later review of the same paper whose first
  3,000 characters match at a `SequenceMatcher` ratio of 0.8 or more.

## Notes

`sample_reviews.parquet` is a fixed sample. 414 of its reviews are not in this file (see `sample_reviews.parquet.md`).

## Read by

`src/panels.py` (`ml`), `src/outcomes.py` (`review_status`), `src/validate.py` (`review_chars`),
`src/between.py` (`claims_table`), `src/external.py` (`intel`), `src/ai.py` (`load_ml_reviews`, `reference_sample`),
`src/figures.py` (`likeness`, `by_year`).
