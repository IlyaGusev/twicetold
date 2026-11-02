# data/clusters_gpt-6-luna-high_other.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the journal panels:
`panels_other.jsonl` (17,520 same-paper panels) and `panels_other_placebo.jsonl` (6,450 control panels). 23,971 rows
for 23,970 panel ids, 582,050 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | `other_<paper id>` or `otherplacebo_<source>_<year>_<n>` |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; only in the 7,131 rows of the second run |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` (absent in the 135 rows with no claims) |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` |
| `error` | Error text, in place of `targets` and `clusters`, when every attempt failed |

16,840 rows come from the first run and 7,131 from the second run. 135 rows have `clusters: []` and no `targets`: in
these panels no review has a claim, so `extract.py` made no call. 5 rows are error rows
(`JSONDecodeError('Expecting value: line 1 column 1 (char 0)')`):

| Panel | Error rows | Usable row |
|---|---|---|
| `other_2112.03919v1` | 2 (one per run) | none |
| `otherplacebo_F1000_2022_117` | 1 (second run) | none |
| `otherplacebo_F1000_2022_31` | 1 (second run) | none |
| `otherplacebo_PLOS_2025_1` | 1 (second run) | none |

## How we got it

1. First run. On 2026-10-31 at 22:03:05 UTC, Claude Code session `881519d6` (project `~/ideas`) started
   `setsid nohup nice -n 19 scripts/extract_all.sh gpt-6-luna-high high high 700` (script of commit `291542f`). After
   the journal pass 1, its `other` chain ran two pass-2 jobs at the same time:
   - `src/extract.py pass2 data/panels_other.jsonl --tag _other --paper_kind general --model gpt-6-luna-high --threads 96 --budget 700 --effort high`
   - `src/extract.py pass2 data/panels_other_placebo.jsonl --tag _other --paper_kind general --model gpt-6-luna-high --threads 48 --budget 700 --effort high`

   The chain finished on 2026-11-01 at 01:52:56 UTC. That `extract.py` did not write an `effort` field. The second
   run overwrote the chain log. No log gives the cost of this run.
2. On 2026-11-01 from 17:25 to 17:50 UTC, `regen.sh` (session `881519d6`, log `logs/regen_gpt-6-luna-high.log`)
   rebuilt the journal tables, `other_pass1.parquet` and both panel files.
3. At 17:51:11 UTC, `driver.sh` ran `reconcile.py` (log `logs/driver_gpt-6-luna-high.log`). It removed 7,145 rows
   (6,450 `otherplacebo_…`, 695 `other_…`) whose panel no longer existed, had another set of reviews or held a review
   with new claims. It first copied the old file (23,985 rows) to
   `/tmp/claude-1000/-home-yallen-ideas/881519d6-0823-4954-a4c3-bf0b4f7ffed9/scratchpad/pre_outputs/`
   (copy in `RAW_ML_REVIEWS_DIR/pre_reconcile_outputs/`).
4. Second run. At 17:55:20 UTC, `driver2.sh` (log `logs/driver2_gpt-6-luna-high.log`) ran
   `nice -n 19 scripts/extract_all.sh gpt-6-luna-high 700`. The code was the working tree between commits `362860d`
   and `6f9fa51`. Log `logs/extract_gpt-6-luna-high/other.log`: `16839 done, 681 to do` (682 calls, 1.05M in, 3.55M
   out, $1.88, `1 items failed`) and `16839 done, 6450 to do` (6,452 calls, 10.39M in, 29.05M out, $15.57,
   `3 items failed`). The dollar values are the `extract.py` estimate from token counts and
   `settings.PRICES_USD_PER_MTOK`. The chain finished at 19:05:26 UTC. Because of the failed items, `extract_all.sh`
   printed `SOME CHAIN FAILED`. The file has no usable row for these 4 panels.
5. Both runs: `extract.py` (`pass2`, `pass2_worker`, `openrouter_call`) listed the claims of
   `claims_gpt-6-luna-high_other.jsonl` for each panel and sent them to OpenRouter `openai/gpt-6-luna` with a strict
   JSON schema, `provider: {"require_parameters": true}` and reasoning effort high. The prompt is
   `src/prompts/pass2.j2` with `paper` = "scientific paper". Its rendered text equals that of
   `prompts/pass2.j2` in commit `291542f`; the changes since then removed a jinja comment and the unused `different` flag.
6. Commit `e03a1c1` (2026-11-01) added the file to Git LFS.

## Reproduce

```
uv run python -I src/extract.py pass2 data/panels_other.jsonl --tag _other --paper_kind general --model gpt-6-luna-high --threads 96 --budget 300 --data <new folder>
uv run python -I src/extract.py pass2 data/panels_other_placebo.jsonl --tag _other --paper_kind general --model gpt-6-luna-high --threads 48 --budget 300 --data <new folder>
```

Pass 2 reads `claims_gpt-6-luna-high_other.jsonl` from the `--data` folder, so run pass 1 there first, or copy this
repo's claims file there. This calls the paid OpenRouter API. Write to a new folder or under a new short name in
`src/settings.py`. Never write into this file: model outputs are never overwritten. `extract.py` appends to its
output and skips panels with a non-error row, so a rerun into this file would try only the 4 failed panels.

## Check

On 2026-11-02 we ran a check script without model calls:

- 23,966 of the 23,970 current panels have a usable row, with the same `review_ids` in the same order as the panel
  file. The 4 panels in the table above have only error rows. No row belongs to a panel outside the two panel files.
- 23,399 usable rows are an exact partition of the panel's claims (631,166 claims in all). 442 rows leave out 579
  claims. 148 rows put 220 claims in two or more places. 29 rows have 47 ids that point to no claim.
- All 135 rows without `targets` belong to panels whose reviews have no claims.
- Every row without `effort` (16,840) is byte-identical to a row in the `pre_outputs` copy of step 3.
- `model` is `gpt-6-luna-high` in all rows. `effort` is `high` or absent; the first run passed `--effort high`. Both
  agree with `settings.REASONING_EFFORTS["gpt-6-luna-high"]`.

## Read by

`src/outcomes.py` `source_panels` (tag `_other`) and `pairs`, through `metrics.load_jsonl` and `metrics.assign`.
`load_jsonl` skips rows with `error`, so the 4 failed panels are not in the outcomes; for a repeated panel id the last
non-error row wins. Both readers skip a row whose `review_ids` differ from the current panel file.
`logs/driver2_gpt-6-luna-high.log` shows `23966 panels, 0 dropped` for `outcomes.py panels --tag _other`.
