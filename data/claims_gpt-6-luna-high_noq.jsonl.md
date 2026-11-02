# data/claims_gpt-6-luna-high_noq.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the reviews without
their questions field, from `noq_reviews.parquet` (4,707 reviews) and `noq_full_reviews.parquet` (42,265 reviews).
One line per review: 46,972 rows, 242,400 claims.

| Field | Meaning |
|---|---|
| `review_id` | OpenReview note id |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; only in the 153 rows of the second run |
| `claims` | List of claims; each claim has `claim`, `category`, `target`, `specific` |

556 reviews have an empty claim list. No row has an `error` field. 46,819 rows come from the first run and 153 from
the second run.

## How we got it

1. First run. On 2026-10-31 at 22:03:05 UTC, Claude Code session `881519d6` (project `~/ideas`) started
   `setsid nohup nice -n 19 scripts/extract_all.sh gpt-6-luna-high high high 700` (script of commit `291542f`). Its
   `noq` chain ran, one after the other:
   - `src/extract.py pass1 data/noq_reviews.parquet --tag _noq --model gpt-6-luna-high --threads 32 --budget 700 --effort high`
   - `src/extract.py pass1 data/noq_full_reviews.parquet --tag _noq --model gpt-6-luna-high --threads 96 --budget 700 --effort high`

   The chain finished on 2026-11-01 at 00:54:47 UTC. That `extract.py` did not write an `effort` field. No log gives
   the cost of this run.
2. On 2026-11-01 from 17:25 to 17:50 UTC, `regen.sh` (session `881519d6`, log `logs/regen_gpt-6-luna-high.log`)
   rebuilt both input files. The reconcile log counts 4,071 new and 4,053 removed reviews in `noq_reviews.parquet`,
   and 4,053 new and 4,071 removed in `noq_full_reviews.parquet`. 153 reviews got a changed text (1 in
   `noq_reviews`, 152 in `noq_full_reviews`).
3. At 17:51:11 UTC, `driver.sh` ran `reconcile.py` (log `logs/driver_gpt-6-luna-high.log`). It removed the 153 rows
   with changed text. It first copied the old file to
   `/tmp/claude-1000/-home-yallen-ideas/881519d6-0823-4954-a4c3-bf0b4f7ffed9/scratchpad/pre_outputs/`
   (copy in `RAW_ML_REVIEWS_DIR/pre_reconcile_outputs/`).
4. Second run. At 17:55:20 UTC, `driver2.sh` (log `logs/driver2_gpt-6-luna-high.log`) ran
   `nice -n 19 scripts/extract_all.sh gpt-6-luna-high 700`. The code was the working tree between commits `362860d`
   and `6f9fa51`. Log `logs/extract_gpt-6-luna-high/noq.log`: `46819 done, 1 to do` then
   `finished: 1 calls, ... $0.00`; `46820 done, 152 to do` then `finished: 152 calls, 0.15M in, 0.17M out, $0.10`. The
   dollar value is the `extract.py` estimate from token counts and `settings.PRICES_USD_PER_MTOK`. The last write was
   at 17:56:16 UTC (file time).
5. Both runs: each review, cut at 14,000 characters, went to OpenRouter `openai/gpt-6-luna` with a strict JSON schema,
   `provider: {"require_parameters": true}` and reasoning effort high (`extract.py` `pass1`, `pass1_worker`,
   `openrouter_call`). The prompt is `src/prompts/pass1.j2` with `paper` = "machine-learning conference paper". The
   template is byte-identical to `prompts/pass1.j2` of commit `291542f`.
6. Commit `e03a1c1` (2026-11-01) added the file to Git LFS.

## Reproduce

```
uv run python -I src/extract.py pass1 data/noq_reviews.parquet --tag _noq --model gpt-6-luna-high --threads 32 --budget 300 --data <new folder>
uv run python -I src/extract.py pass1 data/noq_full_reviews.parquet --tag _noq --model gpt-6-luna-high --threads 96 --budget 300 --data <new folder>
```

This calls the paid OpenRouter API. Write to a new folder (`--data`) or under a new short name in `src/settings.py`.
Never write into this file: model outputs are never overwritten, and `extract.py` appends to its output and skips
done ids. Model output is not deterministic, so a new run gives other claims.

## Check

On 2026-11-02 we ran a check script without model calls:

- The review ids are exactly the 46,972 ids of the two input files together. No id is missing, extra or repeated.
- All 46,972 review ids of `panels_noq.jsonl` and `panels_noq_full.jsonl` have a row.
- Every claim has a string `claim` and a boolean `specific`.
- Every row without `effort` (46,819) is byte-identical to a row in the `pre_outputs` copy of step 3.
- `model` is `gpt-6-luna-high` in all rows. `effort` is `high` or absent; the first run passed `--effort high`. Both
  agree with `settings.REASONING_EFFORTS["gpt-6-luna-high"]`.

## Read by

- `src/outcomes.py` `load_claims`, used by `ml_panels` (tag `_noq`).
- `src/extract.py` `pass2`, as input to pass 2.
