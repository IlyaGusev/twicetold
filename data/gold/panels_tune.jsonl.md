# data/gold/panels_tune.jsonl

The 60 panels of the reference set. Keys: `panel_id`, `review_ids`, `kind`, `year`, `split`.

| Kind | Meaning | dev | test |
|---|---|---|---|
| `same` | 4 reviews of one ICLR paper (from `blind_key.json`), 10 per year | 28 | 12 |
| `cross` | 4 reviews of 4 different ICLR papers of one year (from `panels_placebo.jsonl`), 5 per year | 12 | 8 |

Years: 2018, 2022, 2024, 2026.

## How we got it

All `data/gold/` files except `blind_key.json` come from Claude Code session `881519d6` (project `~/ideas`) on
2026-10-31. That session wrote them to `data/tune/` and `data/` with inline commands; no script was saved. Commit
`d63b8e0` (2026-11-01) moved them to `data/gold/` without changes. `src/gold.py` and `scripts/gold.sh` (2026-11-02)
rebuild the same steps. `tests/test_gold.py` checks that the steps without model calls give byte-identical files.

`src/gold.py sample`:
1. Same-paper panels: every entry of `blind_key.json`, in sorted code order.
2. Cross-paper panels: for each year, `numpy.random.default_rng(7).choice` picks 5 of the sorted
   `placebo_ICLR_<year>_*` panels in `data/panels_placebo.jsonl`. One generator serves all four years in order.
3. Split: in each (kind, year) cell, in panel order, the last 3 same-paper panels or the last 2 cross-paper panels
   are `test`; the rest are `dev`.

Original run: 2026-10-31 14:46 UTC; a second command at 14:47 rewrote the split to the rule in step 3.

## Check

`tests/test_gold.py` rebuilds this file from the inputs above. The result is byte-identical.

## Read by

`src/gold.py` (all later steps), `src/extract.py pass2` (council grouping).
