# data/panels_placebo_area.jsonl

Same-area cross-paper control panels for ICLR. Each panel holds 3 reviews of 3 different ICLR papers of one year
with the same predicted primary area. 2,996 panels with 8,988 review ids (8,276 unique). Keys: `panel_id`
(`placeboA_ICLR_<year>_<k>`), `area` and `review_ids`.

Panels per year: 2017: 297; 2018: 300; 2019: 299; 2020–2026: 300 each. The panels use 21 areas.

## How we got it

1. `uv run python -I src/panels.py robust` (`src/panels.py`, functions `robust` and `robust_full`) reads
   `sample_reviews.parquet` (ICLR rows only), `paper_areas.parquet` and `panels.jsonl`. One run writes
   `panels_placebo_area.jsonl`, `panels_sub3.jsonl`, `panels_noq.jsonl`, `noq_reviews.parquet` (seed 1), then
   `panels_sub3_full.jsonl`, `noq_full_reviews.parquet` and `panels_noq_full.jsonl` (seed 11).
2. Inner function `placebo_area`: the area of a paper is `area_pred` from `paper_areas.parquet` (`other` when
   missing). For each ICLR year, 300 times: pick the area of a random paper, pick 3 papers of that area, then 1 review
   of each paper. It skips a draw when the area has fewer than 3 papers. This gives fewer than 300 panels in 2017
   and 2019.
3. Last write: 2026-11-01 17:27:59 UTC. Step `src/panels.py robust` of `regen.sh` in Claude Code session
   `881519d6` (project `~/ideas`), started at 17:27:54 UTC. The log is `logs/regen_gpt-6-luna-high.log`. It printed
   `noq reviews 4707`, `new sub3 panels 6740` and `new noq reviews 42265 papers 10800`.

## Reproduce

```
uv run python -I src/panels.py robust --out <dir>
```

The inputs (`sample_reviews.parquet`, `paper_areas.parquet`, `panels.jsonl`) come from `data/`. Use `--data <dir>` to
read them from a different folder.

## Check

On 2026-11-02 we ran the "Reproduce" command with the current code and the current inputs in `data/` into a
scratch folder. All seven output files of the run are byte-identical to the files in `data/`.

8,872 of the 8,988 review ids are in the current `reviews.parquet`. The other 116 are ICLR 2017 and 2019 sample
reviews that `corpus.py ml` later dropped as duplicates (see `sample_reviews.parquet.md`).

## Read by

`scripts/extract_all.sh` (pass-2 input of the main chain), `src/outcomes.py` (through the clusters file; the
`placeboA_` prefix gives `placebo_kind = same_area3`).
