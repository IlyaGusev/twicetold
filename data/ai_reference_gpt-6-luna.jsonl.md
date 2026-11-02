# data/ai_reference_gpt-6-luna.jsonl

AI-written reference reviews for the word-use estimate (`ai.py wordshare_fields`). For each sampled human review from
2018–2021, `gpt-6-luna` wrote an outline of its points, then a new review from the outline only. One line per review:
19,194 rows.

| Field | Meaning |
|---|---|
| `review_id` | Id of the human review in `reviews.parquet` or `other_reviews.parquet` |
| `conf`, `year` | Source and year of the human review |
| `outline` | Bullet list from step 1, one `- ` line per point |
| `text` | AI-written review from step 2 |

Rows have no `model` or `effort` field.

| Source | Rows | Years |
|---|---|---|
| BMC-Med | 2,636 | 2018–2021 |
| Copernicus | 2,634 | 2018–2021 |
| F1000 | 3,000 | 2018–2021 |
| ICLR | 3,000 | 2018–2021 |
| NeurIPS | 3,000 | 2021 |
| PLOS | 3,000 | 2019–2021 |
| SciPost | 1,924 | 2018–2021 |

The model is `gpt-6-luna` at reasoning effort low, not `gpt-6-luna-high`. We keep the file as it is: it is a sample
of AI-written text, not a claim extraction.

## How we got it

1. Claude Code session `881519d6` (project `~/ideas`) wrote `src/make_ai_reference.py` on 2026-10-29. The script was
   deleted on 2026-10-30 at 00:01 UTC, after the run. A copy of it, taken at 22:16 UTC during the run, is at
   `/tmp/claude-1000/-home-yallen-ideas/881519d6-0823-4954-a4c3-bf0b4f7ffed9/scratchpad/pre_fire/src/make_ai_reference.py`.
   Its logic now lives in `src/ai.py` (`reference_sample`, `rewrite`, `reference`).
2. Sample: reviews of `reviews.parquet` from 2018–2021 and of `other_reviews.parquet` up to 2021 (the builds of
   2026-10-29), shuffled with `random_state=0`, first 3,000 per source.
3. For each review the script made two calls through `extract.call`: the OpenAI Responses API, model `gpt-6-luna`
   (the default of the `MODEL` variable in the `extract.py` of that time), reasoning effort `low` (the default of
   `extract.call`), strict JSON schema.
   - Outline: the text of `src/prompts/ai_outline.j2`, on the review text cut at 14,000 characters.
   - Review: the text of `src/prompts/ai_write.j2`, with the venue and `words` = the human word count rounded to 10,
     on the outline.

   The old prompt strings and venue names are identical to the current templates and `ai.VENUE`. The script wrote a
   row only when both calls succeeded.
4. Runs, all in session `881519d6` on 2026-10-29 (UTC):
   - 14:17 to 21:21: earlier runs that do not feed this file. The 14:17 run (800 per source, no length target) went
     to `data/ai_reference_v1.jsonl`, which no longer exists. The run started at 18:25 had no API key and wrote no
     rows.
   - 21:21:46: `rm -f data/ai_reference.jsonl; (set -a; . ./.env; set +a; setsid nohup uv run python -I src/make_ai_reference.py > logs/ai_reference2.log 2>&1 &)`.
     The log shows `19194 to write`. The run was stopped at 22:05 UTC. It printed no cost.
   - 22:06:18: `(set -a; . ./.env; set +a; setsid nohup nice -n 19 uv run python -I src/make_ai_reference.py > logs/ai_reference3.log 2>&1 &)`.
     It resumed: `12962 to write`, then `done, $7.63`. The dollar value is the `extract.py` estimate for this run
     only. The last write was at 23:38:13 UTC (file time).

   `logs/ai_reference.log` (`$2.60`) belongs to the first earlier run, not to this file.
5. On 2026-11-01 at 16:52:04 UTC, session `1362ac03` (project `~/twicetold`) renamed `data/ai_reference.jsonl` to
   `data/ai_reference_gpt-6-luna.jsonl`. Commit `e271544` (2026-11-02) added it to Git LFS.

## Reproduce

```
uv run python -I src/ai.py reference --out <new file>
```

This calls the paid OpenAI API with `settings.AI_REFERENCE_MODEL` (`gpt-6-luna`) and
`settings.REASONING_EFFORTS["gpt-6-luna"].pass1` (`low`), the same model and effort as the original run. Always give
a new `--out` path. Model outputs are never overwritten, and `ai.py reference` appends to its output file. The review
tables were rebuilt on 2026-11-01, so the current sample differs (see Check). Model output is not deterministic.

## Check

On 2026-11-02 we ran a check script without model calls:

- 19,194 rows, 19,194 unique review ids. All rows have the same five fields. No `text` or `outline` is empty.
- 18,596 ids are in the current `reviews.parquet` or `other_reviews.parquet`, with the same source and year. 598 ids
  are not (PLOS 565, Copernicus 23, SciPost 10); the 2026-11-01 rebuild removed them.
- `ai.py reference_sample(3000)` on the current tables gives 19,189 ids. 13,752 of them are in this file.
- Word count of the AI review divided by that of the human review: median 0.95 (quartiles 0.87 and 1.03).
- `wordshare_fields` keeps 18,596 rows after its join with the main-sample reviews.

## Read by

`src/ai.py` `wordshare_fields` (default `reference` argument), which writes `reports/llm_share_all.csv`. It joins the
rows to the current reviews on `review_id`, so rows of removed reviews drop out. `src/ai.py reference` resumes from
this file name when called with its default output path.
