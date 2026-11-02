# data/cocodet_input.parquet

The ICLR and NeurIPS reviews that CoCoDet scored. One row per review: 26,484 reviews of 7,000 papers. Columns:
`review_id`, `text` (both strings, copied from `sample_reviews.parquet`). Rows are in `sample_reviews.parquet` order.

The rows are all reviews of a random subset of the panels in `panels.jsonl`:

| Year | ICLR papers | ICLR reviews | NeurIPS papers | NeurIPS reviews |
|---|---|---|---|---|
| 2017 | 300 | 946 | – | – |
| 2018 | 300 | 905 | – | – |
| 2019 | 300 | 916 | – | – |
| 2020 | 300 | 921 | – | – |
| 2021 | 300 | 1,154 | 300 | 1,167 |
| 2022 | 300 | 1,173 | 300 | 1,101 |
| 2023 | 300 | 1,113 | 300 | 1,315 |
| 2024 | 1,000 | 3,862 | 500 | 1,958 |
| 2025 | 1,000 | 4,025 | 500 | 2,026 |
| 2026 | 1,000 | 3,902 | – | – |

## How we got it

No script in the repo makes this file.

1. On 2026-10-29 at 13:14:43 UTC, an inline Python command in Claude Code session `881519d6` (project `~/ideas`)
   wrote it:
   - `rng = np.random.default_rng(7)`;
   - groups the panels of `data/panels.jsonl` by venue and year (venue and year from `papers.parquet`);
   - draws panels without replacement in each group: 300 for ICLR 2017–2023 and NeurIPS 2021–2023, 1,000 for ICLR
     2024–2026, 500 for NeurIPS 2024–2025;
   - keeps the rows of `sample_reviews.parquet` (`review_id`, `text`) whose review is in a drawn panel.
2. The same command block then started CoCoDet on this file (see `cocodet_main.jsonl.md`).

## Reproduce

The original command, with the output path changed:

```
uv run python -I - <<'EOF'
import json, numpy as np, pandas as pd
rng = np.random.default_rng(7)
panels = [json.loads(l) for l in open("data/panels.jsonl")]
P = pd.read_parquet("data/papers.parquet", columns=["paper_id","conf","year"]).set_index("paper_id")
n = {("ICLR",y):300 for y in range(2017,2024)} | {("ICLR",y):1000 for y in (2024,2025,2026)} | {("NeurIPS",y):300 for y in (2021,2022,2023)} | {("NeurIPS",y):500 for y in (2024,2025)}
by = {}
for p in panels:
    m = P.loc[p["panel_id"]]; by.setdefault((m.conf, int(m.year)), []).append(p)
ids = []
for k, ps in by.items():
    for i in rng.choice(len(ps), min(n[k], len(ps)), replace=False): ids += ps[i]["review_ids"]
r = pd.read_parquet("data/sample_reviews.parquet", columns=["review_id","text"])
r[r.review_id.isin(set(ids))].to_parquet("<dir>/cocodet_input.parquet")
EOF
```

## Check

On 2026-11-02 we ran the same steps with the current `panels.jsonl`, `papers.parquet` and `sample_reviews.parquet`
into a scratch folder. The output is identical to this file (`pandas.DataFrame.equals`).

## Read by

`src/ai.py cocodet` (input). No analysis module reads it.
