# data/wd_reviews.parquet

Reviews of withdrawn ICLR 2026 papers. The pass-1 input of the withdrawn-paper check. 3,892 reviews of 1,000 papers.
Columns: `conf` (`ICLR`), `year` (2026), `paper_id`, `review_id`, `text`.

Reviews per paper: 3 (163 papers), 4 (782), 5 (55).

## How we got it

No script in the repo makes this file.

1. On 2026-10-29 at 02:24 UTC, an inline Python command in Claude Code session `1e2ee420` (project `~/twicetold`)
   made it from the `data/reviews.parquet` of that time:
   - ICLR 2026 reviews with `withdrawn` true;
   - papers with 3 or more such reviews (2,186 papers at that time);
   - `numpy.random.default_rng(7)` picks 1,000 of these papers without replacement;
   - the same generator gives each review a random number; the command keeps the first 5 reviews per paper in that
     order;
   - columns `conf`, `year`, `paper_id`, `review_id`, `text` saved.

   The same command wrote the first version of `panels_wd.jsonl`.
2. Last write: 2026-10-29 02:24:14 UTC (file time). The command printed `3892 reviews 1000 withdrawn papers; eligible
   2186`.

The `reviews.parquet` of that time no longer exists. It was older than the copy in `data/backup_predup/` (made
2026-10-29 12:19 UTC and deleted 2026-11-02).

## Reproduce

Not possible with the current data. The code of step 1 is in the transcript of session `1e2ee420` at
2026-10-29 02:24 UTC. It reads `data/reviews.parquet` and writes into `data/`.

## Check

On 2026-11-02 we ran the step-1 code on the current `reviews.parquet` into a scratch folder. The output is not equal:
it has 3,873 reviews, and only 816 review ids are in both files. The current `reviews.parquet` has 5,034 eligible
withdrawn ICLR 2026 papers, not 2,186, so the random draw picks other papers.

Consistency checks against the current `reviews.parquet`:

- 3,891 of the 3,892 reviews are there, with the same text. All 3,891 are ICLR 2026 reviews with `withdrawn` true.
- The missing review is `zlaasankug`. It has the same text as review `foBgDN8Ecm` of the same paper. The duplicate
  (paper, text) filter removed it.

## Read by

`scripts/extract_all.sh` (pass-1 input, tag `_wd`).
