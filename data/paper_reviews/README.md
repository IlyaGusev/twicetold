# data/paper_reviews

LLM reviews written from the paper alone, for the MELD label check (paper Appendix H). 100 ICLR 2020–2022 papers of
`sample_reviews.parquet` (shuffled with seed 0) whose title matches an arXiv entry exactly, after normalization;
205 papers were tried.

| File | Content |
|---|---|
| `papers.jsonl` | `paper_id`, `year`, `title`, `arxiv_id`, `words` (median word count of the paper's human reviews, rounded to 10), `text` (PDF text, whitespace collapsed, first `settings.PAPER_TEXT_MAX_CHARS` characters) |
| `reviews_gpt-6-luna.jsonl`, `reviews_kimi-k3.1-pro.jsonl` | One review per paper: `review_id` (the paper id), `conf`, `year`, `text` |
| `raw/` | Cached arXiv API replies and PDFs; not in git |

## Make

The last steps of `scripts/extract_all.sh`:

```
uv run python -I src/fetch.py arxiv_papers
uv run python -I src/ai.py paper_reviews
```

`arxiv_papers` queries the arXiv API by title and downloads the PDF (network, one request per
`settings.ARXIV_DELAY_SECONDS`). OpenReview serves its PDFs only after a browser challenge, so the step does not use
it. `paper_reviews` sends the paper text with prompt `ai_review` to `gpt-6-luna` (OpenAI) and `kimi-k3.1-pro`
(OpenRouter); paid APIs, output not deterministic.

## Read by

`src/meld.py input` (sets `paper_review`, `paper_review_second`).
