"""Extra panels for robustness checks requested in review.

- panels_placebo_area.jsonl: 3 reviews of 3 different ICLR papers from the same area and year (300 per year).
- panels_sub3.jsonl: a random 3-review subset of ICLR papers with 4-5 reviews, re-clustered from scratch (400 per year).
- panels_noq.jsonl + noq_reviews.parquet: ICLR 2024-2026 papers (400 per year) with the [QUESTIONS] field removed from
  every review; pass 1 is rerun on this text (TAG=_noq).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
rng = np.random.default_rng(1)

sel = pd.read_parquet(DATA / "sample_reviews.parquet")
sel = sel[sel.conf == "ICLR"]
areas = pd.read_parquet(DATA / "paper_areas.parquet")[["paper_id", "area_final"]]
sel = sel.merge(areas, on="paper_id", how="left")
sel["area_final"] = sel.area_final.fillna("other")

with open(DATA / "panels_placebo_area.jsonl", "w") as f:
    for year, g in sel.groupby("year"):
        by_paper = g.groupby("paper_id").review_id.apply(list)
        area_of = g.drop_duplicates("paper_id").set_index("paper_id").area_final
        for k in range(300):
            area = area_of.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
            pool = area_of.index[area_of == area]
            if len(pool) < 3:
                continue
            pids = rng.choice(pool, 3, replace=False)
            rids = [str(rng.choice(by_paper[p])) for p in pids]
            f.write(json.dumps(dict(panel_id=f"placeboA_ICLR_{year}_{k}", area=area, review_ids=rids)) + "\n")

with open(DATA / "panels_sub3.jsonl", "w") as f:
    for year, g in sel.groupby("year"):
        by_paper = g.groupby("paper_id").review_id.apply(list)
        big = by_paper[by_paper.str.len() >= 4]
        for pid in rng.choice(big.index, min(400, len(big)), replace=False):
            rids = [str(r) for r in rng.choice(by_paper[pid], 3, replace=False)]
            f.write(json.dumps(dict(panel_id=f"sub3_{pid}", review_ids=rids)) + "\n")

noq = []
with open(DATA / "panels_noq.jsonl", "w") as f:
    for year in [2024, 2025, 2026]:
        g = sel[sel.year == year]
        pids = rng.choice(g.paper_id.unique(), 400, replace=False)
        g = g[g.paper_id.isin(pids)].copy()
        g["text"] = g.text.str.replace(r"\[QUESTIONS\].*?(?=\[[A-Z]+\]|\Z)", "", regex=True, flags=re.S)
        noq.append(g)
        for pid, gg in g.groupby("paper_id"):
            f.write(json.dumps(dict(panel_id=f"noq_{pid}", review_ids=gg.review_id.tolist())) + "\n")
noq = pd.concat(noq)
noq[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(DATA / "noq_reviews.parquet")
print("noq reviews", len(noq), "chars dropped share",
      1 - noq.text.str.len().sum() / sel[sel.review_id.isin(noq.review_id)].text.str.len().sum())
