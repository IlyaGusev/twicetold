# data/claims_gpt-6-luna-high.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 116,847 reviews of
`sample_reviews.parquet`. One line per review: 116,847 rows, 803,207 claims.

| Field | Meaning |
|---|---|
| `review_id` | OpenReview note id |
| `model` | Short name, always `gpt-6-luna-high` |
| `claims` | List of claims; each claim has `claim` (text), `category` (17 values), `target` (8 values), `specific` (bool) |

1,333 reviews have an empty claim list. No row has an `error` field. No row has an `effort` field.

## How we got it

1. On 2026-10-31 at 22:03:05 UTC, Claude Code session `881519d6` (project `~/ideas`) started
   `setsid nohup nice -n 19 scripts/extract_all.sh gpt-6-luna-high high high 700`. The script was the version of
   commit `291542f`. It gave `--effort high --budget 700` to every pass. This file comes from its first pass:
   `src/extract.py pass1 data/sample_reviews.parquet --model gpt-6-luna-high --threads 256 --budget 700 --effort high`.
2. `extract.py` (`pass1`, `pass1_worker`, `openrouter_call`) sent each review, cut at 14,000 characters, to OpenRouter
   `openai/gpt-6-luna` with a strict JSON schema, `provider: {"require_parameters": true}` and reasoning effort high.
   The prompt is `src/prompts/pass1.j2` with `paper` = "machine-learning conference paper". The template is
   byte-identical to `prompts/pass1.j2` of commit `291542f`. The `extract.py` of that time did not write an `effort`
   field.
3. The last write was on 2026-11-01 at 00:05:36 UTC (file time). The chain logs of this run do not exist any more: the
   rerun of 2026-11-01 17:55 UTC overwrote `logs/extract_gpt-6-luna-high/main.log`.
   `logs/extract_gpt-6-luna-high.launcher.log` keeps only `ALL DONE Sun 01 Nov 2026 02:02:12 AM UTC`. No log gives
   the cost.
4. On 2026-11-01 the review tables were rebuilt. `reconcile.py` (log `logs/driver_gpt-6-luna-high.log`) found no
   changed review text in `sample_reviews.parquet`. The rerun at 17:55 UTC printed `116847 done, 0 to do`. So the
   file did not change after step 3.
5. Commit `e03a1c1` (2026-11-01) added the file to Git LFS.

## Reproduce

```
uv run python -I src/extract.py pass1 data/sample_reviews.parquet --model gpt-6-luna-high --threads 256 --budget 300 --data <new folder>
```

This calls the paid OpenRouter API. Write to a new folder (`--data`) or under a new short name in `src/settings.py`.
Never write into this file: model outputs are never overwritten, and `extract.py` appends to its output and skips
done ids. `scripts/extract_all.sh gpt-6-luna-high` runs all passes, but into `data/`. Model output is not
deterministic, so a new run gives other claims.

## Check

On 2026-11-02 we ran a check script without model calls:

- The review ids are exactly the 116,847 ids of `sample_reviews.parquet`. No id is missing, no id is extra, and no id
  occurs twice.
- All 116,636 review ids of the five pass-2 panel files (`panels.jsonl`, `panels_placebo.jsonl`,
  `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_sub3_full.jsonl`) have a row.
- Every claim has a string `claim` and a boolean `specific`.
- `model` is `gpt-6-luna-high` in all rows. `settings.REASONING_EFFORTS["gpt-6-luna-high"]` is high for pass 1, and
  `settings.OPENROUTER_ROUTES` maps the name to `openai/gpt-6-luna`. Both agree with the launch command.

## Read by

- `src/outcomes.py` `load_claims` (used by `ml_panels` and `pairs`). It keeps rows that have `claims`.
- `src/between.py` `claims_table`, which writes `data/claims_table_gpt-6-luna-high.parquet`. It skips reviews that
  are not in `reviews.parquet`. `between.py embed`, `validate.py` and `figures.py` read that table.
- `src/extract.py` `pass2` (`load_claims`), as input to pass 2.
