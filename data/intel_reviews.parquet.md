# data/intel_reviews.parquet

Human and LLM-written ICLR 2021–2024 reviews from the IntelLabs AI peer-review detection benchmark (test split).
76,729 rows: 38,357 human (`label` 0) and 38,372 LLM (`label` 1).

| Column | Meaning |
|---|---|
| `review_id` | Benchmark review id for a human review; `LLM_<llm>_<review id>` for an LLM review |
| `orig_review_id` | Benchmark `review-id` field (the OpenReview id of the human review) |
| `intel_paper` | Benchmark `PaperId` (a number per year) |
| `label` | 0 = human, 1 = LLM |
| `llm` | Folder of the source file: `claude`, `gemini`, `gpt4o`, `llama`, `qwen` |
| `year` | ICLR year, 2021–2024 |
| `text` | Review fields joined with tags, as in `reviews.parquet` |
| `paper_id` | OpenReview forum id; empty when no human review of the paper is in `reviews.parquet` |

Each `llm` folder holds both labels, so a human review appears once per folder. Rows per year: 2021: 19,192; 2022:
19,432; 2023: 19,068; 2024: 19,037. `paper_id` is set on 57,016 rows (74.3%).

## How we got it

No script in the repo makes this file.

1. On 2026-10-28 at 23:31 UTC, Claude Code session `1e2ee420` (project `~/twicetold`) downloaded the files
   `test/<llm>/ICLR<year>.test.<model>.csv` for 2021–2024 from the Hugging Face dataset
   `IntelLabs/AI-Peer-Review-Detection-Benchmark` with `hf_hub_download` into `/tmp/intel_dl`; a copy is in `RAW_ML_REVIEWS_DIR/intel/`. The download
   metadata gives revision `b80f6524702eb2250139b744430a9a85478586bc`. The models are `claude-sonnet-3.5v2`,
   `gemini-1.5-pro-002`, `gpt-4o`, `llama-3.1-70b` and `qwen2.5-72b`.
2. On 2026-10-28 at 23:36 UTC, an inline Python command in the same session made the file:
   - column names in lower case, spaces to `_`;
   - text: these fields joined in order, whitespace collapsed: `[SUMMARY]` (`summary`, `summary_of_the_paper`),
     `[MAIN]` (`review`, `main_review`), `[STRENGTHS]` (`strengths`, `strength_and_weaknesses`), `[WEAKNESSES]`,
     `[QUESTIONS]`, `[OTHER]` (`clarity,_quality,_novelty_and_reproducibility`, `summary_of_the_review`);
   - keep rows with more than 200 characters of text;
   - `paper_id`: take the human reviews whose `orig_review_id` is in `data/reviews.parquet`, keep the first one per
     `intel_paper` number over all years (`drop_duplicates("intel_paper")`), then map (`year`, `intel_paper`) to its
     `paper_id`.

   The 2023 files name the clarity field `clarity_quality_novelty_and_reproducibility`, without the comma. So that
   field is not in the 2023 texts. The `paper_id` step keeps one year per `intel_paper` number, so a number that
   occurs in two years gets a `paper_id` in one year only: 1,693 (`year`, `intel_paper`) keys have a matched human
   review, but only 1,483 of the 1,992 keys have a `paper_id`.
3. Last write: 2026-10-28 23:36:30 UTC (file time).

On 2026-11-02 we copied `/tmp/intel_dl` to `RAW_ML_REVIEWS_DIR/intel/` (byte-identical).

## Reproduce

```
uv run python -I src/external.py intel --out <dir>/intel_reviews.parquet
```

It reads `RAW_ML_REVIEWS_DIR/intel/test/*/ICLR*.csv` (the step-1 download) and `data/reviews.parquet`.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current `reviews.parquet` into a scratch folder. The output is
identical to this file (`pandas.DataFrame.equals`), `paper_id` included.

## Read by

`src/ai.py` (`wordshare`: LLM reviews; `style`: detector training on 2021–2023, test on 2024), `src/validate.py`
(`detector_placebo`: papers of 2023 and earlier).
