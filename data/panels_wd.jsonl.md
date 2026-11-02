# data/panels_wd.jsonl

Panels of withdrawn ICLR 2026 papers: one panel per paper of `wd_reviews.parquet`. 1,000 panels with 3,891 reviews.
Keys: `panel_id` (`wd_<paper_id>`) and `review_ids`.

Panel sizes: 3 reviews (164), 4 (781), 5 (55).

## How we got it

No script in the repo makes this file.

1. On 2026-10-29 at 02:24 UTC, the inline command that made `wd_reviews.parquet` (Claude Code session `1e2ee420`,
   project `~/twicetold`) wrote the first version: the reviews of `wd_reviews.parquet` grouped by paper, sorted by
   paper id.
2. On 2026-10-29 at 12:19 UTC, `uv run python -I src/dedup_panels.py` in session `881519d6` (project `~/ideas`)
   rewrote the file. The script removed from each panel the reviews whose (paper, text) pair repeats an earlier
   review. It removed 1 review (`zlaasankug`, a copy of `foBgDN8Ecm`) and printed `panels_wd.jsonl: 1 panels
   shortened, 0 dropped, 1000 left`. Before the run, the same command copied the old file to `data/backup_predup/`.
   `src/dedup_panels.py` was never in git; its text is in the session transcript.
3. Last write: 2026-10-29 12:19:18 UTC (file time).

## Reproduce

Not possible with the current data (see `wd_reviews.parquet.md`). The panel step needs only `wd_reviews.parquet`:
group it by `paper_id` in sorted order and drop each review whose (paper, text) pair repeats an earlier row.
This drops only `zlaasankug`.

## Check

On 2026-11-02 we grouped `wd_reviews.parquet` by paper (sorted) and removed every review that has the same text as an
earlier row of the same paper. The result is equal to this file, panel for panel. Also:

- All 3,891 review ids are in `wd_reviews.parquet` and in the current `reviews.parquet`.
- The one review of `wd_reviews.parquet` that is not in a panel is `zlaasankug`.

## Read by

`scripts/extract_all.sh` (pass-2 input, tag `_wd`), `src/outcomes.py` (tag `_wd`, through the clusters file).
