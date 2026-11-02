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

## How we got it

1. On 2026-10-28 at 21:32 UTC, `curl` downloaded four Hugging Face datasets into `RAW_ML_REVIEWS_DIR`
   (`/home/yallen/data/peer-review`). No dataset revision was pinned; each download took the `main` branch.

   | Local file | Source |
   |---|---|
   | `arcade/train-0000{0..5}-of-00006.parquet` | `https://huggingface.co/datasets/ulab-ai/ResearchArcade-openreview-reviews/resolve/main/data/<file>` |
   | `arcade/papers.parquet` | First file under `data/` of `ulab-ai/ResearchArcade-openreview-papers` |
   | `iclr2026/heineman.parquet` | `https://huggingface.co/datasets/davidheineman/iclr-2026/resolve/main/data/main-00000-of-00001.parquet` |
   | `iclr2026/jerma_2026.parquet` | `https://huggingface.co/datasets/JerMa88/ICLR_Peer_Reviews_2026/resolve/main/data/iclr_2026.parquet` |

   The commands are in Claude Code session `881519d6` (project `~/ideas`), 2026-10-28 21:32. The OpenReview API
   blocks anonymous scripts, so we used these dumps.
2. `uv run python -I src/corpus.py ml` builds the table (`src/corpus.py`, functions `from_arcade`, `from_iclr26`,
   `merge_ml`):
   - ResearchArcade gives ICLR 2017–2025 and NeurIPS 2021–2025. It keeps rows whose title starts with
     `Official Review`, and drops EMNLP and ICLR before 2017.
   - ICLR 2026 comes from the Heineman snapshot. The JerMa88 dump adds the questions field, creation and edit
     times, decisions, primary areas and final venues.
   - It keeps reviews with more than 200 characters of text, drops duplicate review ids and duplicate
     (paper, text) pairs.
   - It drops an older version of a review when the same reviewer has a later review of the same paper whose first
     3,000 characters match at a `SequenceMatcher` ratio of 0.8 or more.
3. The last write was on 2026-11-01 between 17:25 and 17:27 UTC, by step `src/corpus.py ml` of
   `regen.sh` in session `881519d6`. The log is `logs/regen_gpt-6-luna-high.log`. The code was the working tree at
   that time, between commits `410e18d` and `6f9fa51`.

An earlier build of this table was in `data/backup_predup/`. `sample_reviews.parquet` was drawn from it. That folder
was deleted on 2026-11-02 and no copy exists.

## Reproduce

```
uv run python -I src/corpus.py ml --out <dir>
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the raw dumps in `RAW_ML_REVIEWS_DIR` into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`).

## Read by

`src/panels.py` (`ml`), `src/outcomes.py`, `src/validate.py` (`review_chars`), `src/between.py` (`claims_table`), `src/ai.py`, `src/figures.py`
(corpus table).
