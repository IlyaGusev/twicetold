# data/clusters_gpt-6-luna-high_icml.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the 3,000 ICML panels
of `panels_icml.jsonl`. One line per panel: 3,000 rows, 65,070 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | `icml_<paper id>` |
| `model` | Short name, always `gpt-6-luna-high` |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` |

No row has an `error` or `effort` field. No panel id occurs twice.

## How we got it

1. On 2026-10-31 at 22:03:05 UTC, Claude Code session `881519d6` (project `~/ideas`) started
   `setsid nohup nice -n 19 scripts/extract_all.sh gpt-6-luna-high high high 700` (script of commit `291542f`). After
   the ICML pass 1 (last write 22:51:54 UTC), its `icml` chain ran
   `src/extract.py pass2 data/panels_icml.jsonl --tag _icml --model gpt-6-luna-high --threads 64 --budget 700 --effort high`.
   The chain finished at 23:23:42 UTC, which is also the file time.
2. `extract.py` (`pass2`, `pass2_worker`, `openrouter_call`) listed the claims of
   `claims_gpt-6-luna-high_icml.jsonl` for each panel and sent them to OpenRouter `openai/gpt-6-luna` with a strict
   JSON schema, `provider: {"require_parameters": true}` and reasoning effort high. The prompt is
   `src/prompts/pass2.j2` with `paper` = "machine-learning paper" and `different` = false. Its rendered text equals that
   of `prompts/pass2.j2` in commit `291542f`; the only change since then removed a jinja comment. That `extract.py`
   did not write an `effort` field.
3. The rerun of 2026-11-01 17:55 UTC overwrote the chain log `logs/extract_gpt-6-luna-high/icml.log`; the new log
   shows `3000 done, 0 to do`. `reconcile.py` found 0 stale rows. So the file did not change after step 1. No log
   gives the cost of the first run.
4. Commit `e03a1c1` (2026-11-01) added the file to Git LFS.

## Reproduce

```
uv run python -I src/extract.py pass2 data/panels_icml.jsonl --tag _icml --model gpt-6-luna-high --threads 64 --budget 300 --data <new folder>
```

Pass 2 reads `claims_gpt-6-luna-high_icml.jsonl` from the `--data` folder, so run pass 1 there first, or copy this
repo's claims file there. This calls the paid OpenRouter API. Write to a new folder or under a new short name in
`src/settings.py`. Never write into this file: model outputs are never overwritten, and `extract.py` appends to its
output and skips done panels. Model output is not deterministic.

## Check

On 2026-11-02 we ran a check script without model calls:

- Every panel of `panels_icml.jsonl` has one row, with the same `review_ids` in the same order.
- 2,950 rows are an exact partition of the panel's claims (74,693 claims in all). 39 rows leave out one claim each.
  11 rows put one claim each in two places. 1 row has 1 id that points to no claim.
- `model` is `gpt-6-luna-high` in all rows. The launch command used effort high, which agrees with
  `settings.REASONING_EFFORTS["gpt-6-luna-high"]`.

## Read by

`src/outcomes.py` `source_panels` (tag `_icml`) and `pairs`, through `metrics.load_jsonl` (skips rows with `error`;
for a repeated panel id the last row wins) and `metrics.assign`. Both skip a row whose `review_ids` differ from
the current panel file.
