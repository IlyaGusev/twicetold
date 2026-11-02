# data/clusters_gpt-6-luna-high_noq.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the panels of reviews
without their questions field: `panels_noq.jsonl` (1,200 panels) and `panels_noq_full.jsonl` (10,800 panels). One line
per panel: 12,000 rows, 209,515 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | `noq_<paper id>` |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; only in the 151 rows of the second run |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` |

No row has an `error` field. No panel id occurs twice. 11,849 rows come from the first run and 151 from the second
run.

## How we got it

1. First run. On 2026-10-31 at 22:03:05 UTC, Claude Code session `881519d6` (project `~/ideas`) started
   `setsid nohup nice -n 19 scripts/extract_all.sh gpt-6-luna-high high high 700` (script of commit `291542f`). After
   both noq pass-1 jobs, its `noq` chain ran two pass-2 jobs at the same time:
   - `src/extract.py pass2 data/panels_noq.jsonl --tag _noq --model gpt-6-luna-high --threads 32 --budget 700 --effort high`
   - `src/extract.py pass2 data/panels_noq_full.jsonl --tag _noq --model gpt-6-luna-high --threads 96 --budget 700 --effort high`

   The chain finished on 2026-11-01 at 00:54:47 UTC. That `extract.py` did not write an `effort` field. The second
   run overwrote the chain log. No log gives the cost of this run.
2. On 2026-11-01 from 17:25 to 17:50 UTC, `regen.sh` (session `881519d6`, log `logs/regen_gpt-6-luna-high.log`)
   rebuilt the noq inputs and panels. 153 reviews got a changed text, and their claims were extracted again.
3. At 17:51:11 UTC, `driver.sh` ran `reconcile.py` (log `logs/driver_gpt-6-luna-high.log`). It removed 151 `noq_…`
   rows whose panel had another set of reviews or held a review with new claims. It first copied the old file to
   `/tmp/claude-1000/-home-yallen-ideas/881519d6-0823-4954-a4c3-bf0b4f7ffed9/scratchpad/pre_outputs/`.
4. Second run. At 17:55:20 UTC, `driver2.sh` (log `logs/driver2_gpt-6-luna-high.log`) ran
   `nice -n 19 scripts/extract_all.sh gpt-6-luna-high 700`. The code was the working tree between commits `362860d`
   and `6f9fa51`. Log `logs/extract_gpt-6-luna-high/noq.log`: `11849 done, 1 to do` (1 call, $0.00) and
   `11849 done, 150 to do` (150 calls, 0.18M in, 0.61M out, $0.32). The dollar values are the `extract.py` estimate
   from token counts and `settings.PRICES_USD_PER_MTOK`. The last write was at 17:57:44 UTC (file time).
5. Both runs: `extract.py` (`pass2`, `pass2_worker`, `openrouter_call`) listed the claims of
   `claims_gpt-6-luna-high_noq.jsonl` for each panel and sent them to OpenRouter `openai/gpt-6-luna` with a strict
   JSON schema, `provider: {"require_parameters": true}` and reasoning effort high. The prompt is
   `src/prompts/pass2.j2` with `paper` = "machine-learning paper" and `different` = false. Its rendered text equals that
   of `prompts/pass2.j2` in commit `291542f`; the only change since then removed a jinja comment.
6. Commit `e03a1c1` (2026-11-01) added the file to Git LFS.

## Reproduce

```
uv run python -I src/extract.py pass2 data/panels_noq.jsonl --tag _noq --model gpt-6-luna-high --threads 32 --budget 300 --data <new folder>
uv run python -I src/extract.py pass2 data/panels_noq_full.jsonl --tag _noq --model gpt-6-luna-high --threads 96 --budget 300 --data <new folder>
```

Pass 2 reads `claims_gpt-6-luna-high_noq.jsonl` from the `--data` folder, so run pass 1 there first, or copy this
repo's claims file there. This calls the paid OpenRouter API. Write to a new folder or under a new short name in
`src/settings.py`. Never write into this file: model outputs are never overwritten, and `extract.py` appends to its
output and skips done panels. Model output is not deterministic.

## Check

On 2026-11-02 we ran a check script without model calls:

- Every panel of the two panel files has one row, with the same `review_ids` in the same order.
- 11,882 rows are an exact partition of the panel's claims (242,400 claims in all). 96 rows leave out 99 claims. 23
  rows put 24 claims in two places. No id points to no claim.
- Every row without `effort` (11,849) is byte-identical to a row in the `pre_outputs` copy of step 3.
- `model` is `gpt-6-luna-high` in all rows. `effort` is `high` or absent; the first run passed `--effort high`. Both
  agree with `settings.REASONING_EFFORTS["gpt-6-luna-high"]`.

## Read by

`src/outcomes.py` `ml_panels` (tag `_noq`), through `metrics.load_jsonl` (skips rows with `error`; for a repeated
panel id the last row wins) and `metrics.assign`.
