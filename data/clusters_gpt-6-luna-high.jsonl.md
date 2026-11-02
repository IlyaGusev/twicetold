# data/clusters_gpt-6-luna-high.jsonl

Pass-2 grouping of `gpt-6-luna-high` (OpenRouter `openai/gpt-6-luna`, reasoning effort high) for the main ML panels and
their controls. One line per panel: 47,979 rows, 1,047,327 clusters.

| Field | Meaning |
|---|---|
| `panel_id` | Panel id: a paper id, `placebo_…`, `placeboA_…` or `sub3_…` |
| `model` | Short name, always `gpt-6-luna-high` |
| `effort` | `high`; only in the 8,499 rows of the second run |
| `review_ids` | Reviews of the panel, in the order the prompt used |
| `targets` | Elements of the paper: `target_id`, `element` |
| `clusters` | Problem groups: `target_id`, `problem`, `claim_ids` such as `R2-5` (review 2 of `review_ids`, claim 5) |

No row has an `error` field. No panel id occurs twice.

| Panel file | Rows | First run | Second run |
|---|---|---|---|
| `panels.jsonl` | 30,930 | 30,927 | 3 |
| `panels_placebo.jsonl` | 4,500 | 4,500 | 0 |
| `panels_placebo_area.jsonl` | 2,996 | 1,796 | 1,200 |
| `panels_sub3.jsonl` | 2,773 | 311 | 2,462 |
| `panels_sub3_full.jsonl` | 6,740 | 1,906 | 4,834 |
| `panels_blind.jsonl` (ids `B00`–`B39`; the file no longer exists) | 40 | 40 | 0 |

## How we got it

1. First run. On 2026-10-31 at 22:03:05 UTC, Claude Code session `881519d6` (project `~/ideas`) started
   `setsid nohup nice -n 19 scripts/extract_all.sh gpt-6-luna-high high high 700` (script of commit `291542f`). After
   the main pass 1 (last write 2026-11-01 00:05:36 UTC), its `main` chain ran six pass-2 jobs at the same time, each
   `src/extract.py pass2 <panel file> --model gpt-6-luna-high --budget 700 --effort high`: `panels.jsonl` (160
   threads), `panels_placebo.jsonl`, `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_sub3_full.jsonl` (32
   threads each) and `panels_blind.jsonl` (8 threads). The chain finished on 2026-11-01 at 02:02:12 UTC. That
   `extract.py` did not write an `effort` field. The second run overwrote the chain log. No log gives the cost of
   this run.
2. On 2026-11-01 from 17:25 to 17:50 UTC, `regen.sh` (session `881519d6`, log `logs/regen_gpt-6-luna-high.log`)
   rebuilt the review tables and the panel files.
3. At 17:51:11 UTC, `driver.sh` ran `reconcile.py` (log `logs/driver_gpt-6-luna-high.log`). It removed 8,504 rows
   whose panel no longer existed or had another set of reviews: 7,303 `sub3_…`, 1,198 `placeboA_…` and 3 paper
   panels. It first copied the old file (47,984 rows) to
   `/tmp/claude-1000/-home-yallen-ideas/881519d6-0823-4954-a4c3-bf0b4f7ffed9/scratchpad/pre_outputs/`
   (copy in `RAW_ML_REVIEWS_DIR/pre_reconcile_outputs/`). Its guard then
   stopped the driver (15,800 stale panels in all files, limit 12,000), but the removals stayed.
4. Second run. At 17:55:20 UTC, `driver2.sh` (log `logs/driver2_gpt-6-luna-high.log`) ran
   `nice -n 19 scripts/extract_all.sh gpt-6-luna-high 700`. The code was the working tree between commits `362860d`
   and `6f9fa51`. Each pass printed `39480 done`. Log `logs/extract_gpt-6-luna-high/main.log` gives `3 to do` (3
   calls, $0.00), `1200 to do` (1,200 calls, $2.34), `2462 to do` (2,465 calls, $5.49) and `4834 to do` (4,834 calls,
   $10.66). The dollar values are the `extract.py` estimate from token counts and `settings.PRICES_USD_PER_MTOK`. The
   chain finished at 19:07:33 UTC.
5. Both runs: `extract.py` (`pass2`, `pass2_worker`, `openrouter_call`) listed the claims of
   `claims_gpt-6-luna-high.jsonl` for each panel as `R<review>-<claim> [category; target]: text` and sent them to
   OpenRouter `openai/gpt-6-luna` with a strict JSON schema, `provider: {"require_parameters": true}` and reasoning
   effort high. The prompt is `src/prompts/pass2.j2` with `paper` = "machine-learning paper".
   Its rendered text equals that of `prompts/pass2.j2` in commit `291542f`; the changes since then removed a jinja comment and the unused `different` flag.
6. Commit `e03a1c1` (2026-11-01) added the file to Git LFS.

## Reproduce

```
uv run python -I src/extract.py pass2 data/panels.jsonl --model gpt-6-luna-high --threads 160 --budget 300 --data <new folder>
```

Run the same command for `panels_placebo.jsonl`, `panels_placebo_area.jsonl`, `panels_sub3.jsonl` and
`panels_sub3_full.jsonl` (32 threads each). Pass 2 reads `claims_gpt-6-luna-high.jsonl` from the `--data` folder, so
run pass 1 there first, or copy this repo's claims file there. This calls the paid OpenRouter API. Write to a new
folder or under a new short name in `src/settings.py`. Never write into this file: model outputs are never
overwritten, and `extract.py` appends to its output and skips done panels. Model output is not deterministic.

## Check

On 2026-11-02 we ran a check script without model calls:

- Every panel of the five current panel files has one row. The 40 `B…` rows match no current panel file.
- 1,516 `sub3_…` rows (263 from `panels_sub3.jsonl`, 1,253 from `panels_sub3_full.jsonl`, all from the first run)
  list the same reviews as the panel file, in another order. Readers use the order in the row, so the claim ids still
  point to the claims the model saw. `reconcile.py` compared sorted lists, so it kept these rows.
- For the other 46,423 rows, we checked the claim ids against the claims file. 45,723 rows are an exact partition of
  the panel's 1,146,949 claims. 560 rows leave out 593 claims. 144 rows put 177 claims in two or more places. 19 rows
  have 22 ids that point to no claim. Of the 1,516 reordered rows, 1,502 are an exact partition.
- Every row without `effort` (39,480) is byte-identical to a row in the `pre_outputs` copy of step 3.
- `model` is `gpt-6-luna-high` in all rows. `effort` is `high` or absent; the first run passed `--effort high`. Both
  agree with `settings.REASONING_EFFORTS["gpt-6-luna-high"]`.

## Read by

`src/outcomes.py` `ml_panels` (tag `""`) and `pairs`, through `metrics.load_jsonl` (skips rows with `error`; for a
repeated panel id the last row wins) and `metrics.assign`. `assign` puts a left-out claim in its own cluster, keeps the
first cluster of a claim that occurs twice, and ignores ids that point to no claim. `ml_panels` and `pairs` skip the
`B…` rows, because these ids are not papers.
