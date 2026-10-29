"""Full-scale versions of the robustness variants: re-clustered 3-review subsets for ALL sampled ICLR 2023 and 2026
papers with 4+ reviews (panels_sub3_full.jsonl), and the no-questions rerun for ALL sampled ICLR 2024-2026 papers
(noq_full_reviews.parquet, panels_noq_full.jsonl). Panels already covered by the 400-per-year files are skipped."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
rng = np.random.default_rng(11)
sel = pd.read_parquet(DATA / "sample_reviews.parquet")
sel = sel[sel.conf == "ICLR"]

done_sub3 = {json.loads(l)["panel_id"] for l in open(DATA / "panels_sub3.jsonl")}
n = 0
with open(DATA / "panels_sub3_full.jsonl", "w") as f:
    for year in [2023, 2026]:
        by_paper = sel[sel.year == year].groupby("paper_id").review_id.apply(list)
        for pid, rids in by_paper[by_paper.str.len() >= 4].items():
            if f"sub3_{pid}" in done_sub3:
                continue
            f.write(json.dumps(dict(panel_id=f"sub3_{pid}", review_ids=[str(r) for r in rng.choice(rids, 3, replace=False)])) + "\n"); n += 1
print("new sub3 panels", n)

done_noq = {json.loads(l)["panel_id"] for l in open(DATA / "panels_noq.jsonl")}
g = sel[sel.year.isin([2024, 2025, 2026]) & ~sel.paper_id.map(lambda p: f"noq_{p}" in done_noq)].copy()
g["text"] = g.text.str.replace(r"\[QUESTIONS\].*?(?=\[[A-Z]+\]|\Z)", "", regex=True, flags=re.S)
g[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(DATA / "noq_full_reviews.parquet")
with open(DATA / "panels_noq_full.jsonl", "w") as f:
    for pid, gg in g.groupby("paper_id"):
        f.write(json.dumps(dict(panel_id=f"noq_{pid}", review_ids=gg.review_id.tolist())) + "\n")
print("new noq reviews", len(g), "papers", g.paper_id.nunique())
