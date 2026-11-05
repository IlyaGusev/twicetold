# data/gold_audit

The audit extension of the council reference (paper Appendix C): 40 same-paper panels of 4 reviews (fewer for some
journal panels), made with the same council steps as `data/gold`.

| Stratum | Panels | Sample |
|---|---|---|
| `same2023` | 10 | ICLR 2023 papers with at least 4 reviews; 4 random reviews |
| `flag` | 15 | ICLR 2024–2026 papers with at least 2 flagged and 2 unflagged reviews; 2 of each |
| `journal` | 15 | 3 per journal source from 2024 on, with at least 2 flagged and 1 unflagged report; up to 2 unflagged |

`panels_tune.jsonl` adds `stratum`, `source`, `paper_id` and `paper_kind` to the fields of `data/gold/panels_tune.jsonl`;
`split` is `audit`. `panels_tune_<kind>.jsonl` and `tune_reviews_<kind>.parquet` split the panels by prompt wording.
`claims_gpt-6-luna-high.jsonl` copies the main Luna claims of these reviews; `clusters_gpt-6-luna-high.jsonl` holds
Luna groups of these exact panels. `match/` holds the Opus 5.5 claim match.

## Make

```
scripts/gold.sh data/gold_audit 25 sample_audit
```

`sample_audit` is deterministic (seed 11). The other steps call paid APIs. In this run, one claim judgment failed in
all attempts; `judge_claims` was run again before the grouping steps.

## Read by

`src/validate.py audit`.
