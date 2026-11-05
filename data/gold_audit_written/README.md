# data/gold_audit_written

The AI-written extension of the council reference (paper Appendix C): same-paper panels of 4 ICLR 2024–2026 reviews,
2 that MELD labels AI-written and 2 that it does not flag, made with the same council steps as `data/gold_audit`.

| File | Content |
|---|---|
| `panels_tune.jsonl`, `panels_tune_<kind>.jsonl` | 58 panels, fields as in `data/gold_audit`; `stratum` is `written` |
| `tune_reviews.parquet`, `tune_reviews_<kind>.parquet` | Texts of the 240 sampled reviews |
| `claims_council.jsonl`, `adjudicate/`, `clusters_council.jsonl`, `adjudicate2/` | Council claims, adjudicated claims, council groups and adjudicated groups, as in `data/gold` |
| `claims_gpt-6-luna-high.jsonl` | Main Luna claims of these reviews |
| `clusters_gpt-6-luna-high.jsonl` | Luna groups of these exact panels |
| `match/` | Opus 5.5 claim match |

## Make

```
scripts/gold.sh data/gold_audit_written 25 sample_written
```

`sample_written` draws 60 papers (seed 12) in a shuffled order of ICLR 2024–2026 papers with at least 2 AI-written and
2 unflagged reviews. Opus 5.5 returned no claims for the 4 reviews of panel `A15` in three runs; `claim_inputs` drops
panels with no claims from a council member, and `group_inputs` drops panels with no adjudicated claims (one review of one panel failed in every judge attempt), so 58 panels remain. The other steps call paid APIs.

## Read by

`src/validate.py audit`.
