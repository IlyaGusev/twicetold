# data/claims_gpt-6-luna-high_other.jsonl

Pass-1 claims of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the journal reviews of
`other_pass1.parquet` (SciPost, F1000, BMC Medicine, PLOS ONE, Copernicus). One line per review: 42,306 rows,
443,134 claims.

| Field | Meaning |
|---|---|
| `review_id` | Review id from the journal source |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; only in the 1,356 rows of the second run |
| `claims` | List of claims; each claim has `claim`, `category`, `target`, `specific` |

2,147 reviews have an empty claim list. No row has an `error` field. 40,950 rows come from the first run and 1,356
from the second run. 1,314 rows belong to reviews that the 2026-11-01 rebuild removed from `other_pass1.parquet`.

## How we got it

1. First run. On 2026-10-31 at 22:03:05 UTC, Claude Code session `881519d6` (project `~/ideas`) started
   `setsid nohup nice -n 19 scripts/extract_all.sh gpt-6-luna-high high high 700` (script of commit `291542f`). Its
   `other` chain ran
   `src/extract.py pass1 data/other_pass1.parquet --tag _other --paper_kind general --model gpt-6-luna-high --threads 96 --budget 700 --effort high`.
   The chain finished on 2026-11-01 at 01:52:56 UTC. That `extract.py` did not write an `effort` field. No log gives
   the cost of this run.
2. On 2026-11-01 from 17:25 to 17:50 UTC, `regen.sh` (session `881519d6`, log `logs/regen_gpt-6-luna-high.log`)
   rebuilt the journal tables and `other_pass1.parquet`: 1,263 new reviews, 93 reviews with changed text, 1,314
   reviews removed.
3. At 17:51:11 UTC, `driver.sh` ran `reconcile.py` (log `logs/driver_gpt-6-luna-high.log`). It removed the 93 rows
   with changed text. It kept the rows of the 1,314 removed reviews. It first copied the old file to
   `/tmp/claude-1000/-home-yallen-ideas/881519d6-0823-4954-a4c3-bf0b4f7ffed9/scratchpad/pre_outputs/`
   (copy in `RAW_ML_REVIEWS_DIR/pre_reconcile_outputs/`).
4. Second run. At 17:55:20 UTC, `driver2.sh` (log `logs/driver2_gpt-6-luna-high.log`) ran
   `nice -n 19 scripts/extract_all.sh gpt-6-luna-high 700`. The code was the working tree between commits `362860d`
   and `6f9fa51`. The pass printed `40950 done, 1356 to do` and
   `finished: 1356 calls, 1.75M in, 3.30M out, $1.83` (log `logs/extract_gpt-6-luna-high/other.log`). The dollar
   value is the `extract.py` estimate from token counts and `settings.PRICES_USD_PER_MTOK`. The last write was at
   18:00:34 UTC (file time).
5. Both runs: each review, cut at 14,000 characters, went to OpenRouter `openai/gpt-6-luna` with a strict JSON schema,
   `provider: {"require_parameters": true}` and reasoning effort high (`extract.py` `pass1`, `pass1_worker`,
   `openrouter_call`). The prompt is `src/prompts/pass1.j2` with `paper` = "scientific paper submitted to a journal".
   The template is byte-identical to `prompts/pass1.j2` of commit `291542f`, and the wording table had the same
   text.
6. Commit `e03a1c1` (2026-11-01) added the file to Git LFS.

## Reproduce

```
uv run python -I src/extract.py pass1 data/other_pass1.parquet --tag _other --paper_kind general --model gpt-6-luna-high --threads 96 --budget 300 --data <new folder>
```

This calls the paid OpenRouter API. Write to a new folder (`--data`) or under a new short name in `src/settings.py`.
Never write into this file: model outputs are never overwritten, and `extract.py` appends to its output and skips
done ids. Model output is not deterministic, so a new run gives other claims.

## Check

On 2026-11-02 we ran a check script without model calls:

- All 40,992 review ids of `other_pass1.parquet` have a row. No id occurs twice. 1,314 ids are not in
  `other_pass1.parquet` (step 3).
- All 40,992 review ids of `panels_other.jsonl` and `panels_other_placebo.jsonl` have a row. No panel uses the 1,314
  extra rows.
- Every claim has a string `claim` and a boolean `specific`.
- Every row without `effort` (40,950) is byte-identical to a row in the `pre_outputs` copy of step 3.
- `model` is `gpt-6-luna-high` in all rows. `effort` is `high` or absent; the first run passed `--effort high`. Both
  agree with `settings.REASONING_EFFORTS["gpt-6-luna-high"]`.

## Read by

- `src/outcomes.py` `load_claims`, used by `source_panels` (tag `_other`) and `pairs`.
- `src/extract.py` `pass2`, as input to pass 2.
