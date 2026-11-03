# data/match_gpt-6-luna-high/claims.jsonl

Input of the claim match between `gpt-6-luna-high` and the council reference. 480 lines: for each of the 240 reviews
of `data/gold/panels_tune.jsonl`, one line `ref_<review_id>` with the reference claims (`data/gold/adjudicate/out.jsonl`)
and one line `<review_id>` with the Luna claims (`data/claims_gpt-6-luna-high.jsonl`). Fields: `review_id`, `claims`.

## Make

```
uv run python -I src/gold.py match_claims --model gpt-6-luna-high
```

Deterministic, no API; the same command then runs the paid match (`clusters_opus-5.5.jsonl`). `scripts/extract_all.sh`
runs it after `merge`.

## Notes

- Git LFS stores it.

## Read by

`src/extract.py pass2` (as `--claims_file`), called by `src/gold.py match_claims`.
