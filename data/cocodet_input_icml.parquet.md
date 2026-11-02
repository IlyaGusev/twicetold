# data/cocodet_input_icml.parquet

The ICML reviews that CoCoDet scored: all of `icml_sample.parquet`. One row per review: 11,500 rows (ICML 2025:
5,713; ICML 2026: 5,787). Columns: `review_id`, `text` (strings). Rows are in `icml_sample.parquet` order.

The texts keep the ICML field tags, for example `[CLAIMS AND EVIDENCE]`. All 11,500 rows have at least one tag that
the CoCoDet script of that time did not remove (see `cocodet_icml.jsonl.md`).

## How we got it

No script in the repo makes this file.

On 2026-10-29 at 18:23:03 UTC, Claude Code session `881519d6` (project `~/ideas`) ran:

```
uv run python -I -c "import pandas as pd; pd.read_parquet('data/icml_sample.parquet', columns=['review_id','text']).to_parquet('data/cocodet_input_icml.parquet')"
```

The same command line then started CoCoDet on this file.

## Reproduce

```
uv run python -I -c "import pandas as pd; pd.read_parquet('data/icml_sample.parquet', columns=['review_id','text']).to_parquet('<dir>/cocodet_input_icml.parquet')"
```

## Check

On 2026-11-02 we ran the "Reproduce" command with the current `icml_sample.parquet` into a scratch folder. The output
is identical to this file (`pandas.DataFrame.equals`).

## Read by

`src/ai.py cocodet` (input). No analysis module reads it.
