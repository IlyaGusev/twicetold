# data/intel_reviews.parquet

Human and LLM-written ICLR 2021–2024 reviews from the IntelLabs AI peer-review detection benchmark (test split). One
row per review: 76,729 rows, 38,357 human (`label` 0) and 38,372 LLM (`label` 1).

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

Each `llm` folder holds both labels, so a human review occurs once per folder. Rows per year: 2021: 19,192; 2022:
19,432; 2023: 19,068; 2024: 19,037. `paper_id` is set on 57,016 rows (74.3%).

## Make

```
uv run python -I src/external.py intel --out data/intel_reviews.parquet
```

No run chain calls it. Deterministic, no API, no network. It reads
`settings.INTEL_DIR/test/*/ICLR*.csv` and `data/reviews.parquet`, and keeps rows with more than 200 characters of
text. The CSVs are a saved download of the Hugging Face dataset `IntelLabs/AI-Peer-Review-Detection-Benchmark`
(revision `b80f6524702eb2250139b744430a9a85478586bc`). No code in the repo downloads them. The LLMs are
`claude-sonnet-3.5v2`, `gemini-1.5-pro-002`, `gpt-4o`, `llama-3.1-70b` and `qwen2.5-72b`.

## Notes

- The 2023 CSVs name the clarity field `clarity_quality_novelty_and_reproducibility`, without the comma. So the
  2023 texts do not have that field.
- `paper_id` comes from the first matched human review per `intel_paper` number over all years. A number that
  occurs in two years gets a `paper_id` in one year only: 1,483 of the 1,992 (`year`, `intel_paper`) keys have one.
- Git LFS stores it.

## Read by

`src/meld.py input` (1,000 LLM reviews per LLM, the `intel_ai` recall set).
