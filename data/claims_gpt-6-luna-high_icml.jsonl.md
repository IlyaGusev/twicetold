# data/claims_gpt-6-luna-high_icml.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 11,500 ICML reviews
of `icml_pass1.parquet`. One line per review: 11,500 rows, 74,693 claims.

| Field | Meaning |
|---|---|
| `review_id` | OpenReview note id |
| `model` | Short name, always `gpt-6-luna-high` |
| `claims` | List of claims; each claim has `claim`, `category`, `target`, `specific` |

168 reviews have an empty claim list. No row has an `error` field. No row has an `effort` field.

## How we got it

1. On 2026-10-31 at 22:03:05 UTC, Claude Code session `881519d6` (project `~/ideas`) started
   `setsid nohup nice -n 19 scripts/extract_all.sh gpt-6-luna-high high high 700`. The script was the version of
   commit `291542f`. This file comes from the first pass of its `icml` chain:
   `src/extract.py pass1 data/icml_pass1.parquet --tag _icml --model gpt-6-luna-high --threads 64 --budget 700 --effort high`.
2. `extract.py` (`pass1`, `pass1_worker`, `openrouter_call`) sent each review, cut at 14,000 characters, to OpenRouter
   `openai/gpt-6-luna` with a strict JSON schema, `provider: {"require_parameters": true}` and reasoning effort high.
   The prompt is `src/prompts/pass1.j2` with `paper` = "machine-learning conference paper". The template is
   byte-identical to `prompts/pass1.j2` of commit `291542f`. The `extract.py` of that time did not write an `effort`
   field.
3. The last write was on 2026-10-31 at 22:51:54 UTC (file time). The rerun of 2026-11-01 17:55 UTC overwrote the
   chain log `logs/extract_gpt-6-luna-high/icml.log`; the new log shows `11500 done, 0 to do`. So the file did not
   change after the first run. No log gives the cost of the first run.
4. Commit `e03a1c1` (2026-11-01) added the file to Git LFS.

## Reproduce

```
uv run python -I src/extract.py pass1 data/icml_pass1.parquet --tag _icml --model gpt-6-luna-high --threads 64 --budget 300 --data <new folder>
```

This calls the paid OpenRouter API. Write to a new folder (`--data`) or under a new short name in `src/settings.py`.
Never write into this file: model outputs are never overwritten, and `extract.py` appends to its output and skips
done ids. Model output is not deterministic, so a new run gives other claims.

## Check

On 2026-11-02 we ran a check script without model calls:

- The review ids are exactly the 11,500 ids of `icml_pass1.parquet`. No id is missing, extra or repeated.
- All 11,500 review ids of `panels_icml.jsonl` have a row.
- Every claim has a string `claim` and a boolean `specific`.
- `model` is `gpt-6-luna-high` in all rows. The launch command used effort high, which agrees with
  `settings.REASONING_EFFORTS["gpt-6-luna-high"]`.

## Read by

- `src/outcomes.py` `load_claims`, used by `source_panels` (tag `_icml`) and `pairs`.
- `src/extract.py` `pass2`, as input to pass 2.
