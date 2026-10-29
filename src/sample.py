"""Choose the papers and reviews to annotate, and build the real and placebo panels.

Main corpus: all ICLR 2017-2023 non-withdrawn papers with 3+ reviews, 3,000 each from ICLR 2024 and 2025, 6,000 from
ICLR 2026, and 1,000 per year from NeurIPS 2021-2025. At most 5 reviews per paper (random). Placebo panels take 4 reviews
from 4 different papers of the same venue-year.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
QUOTA = {("ICLR", 2024): 3000, ("ICLR", 2025): 3000, ("ICLR", 2026): 6000,
         **{("NeurIPS", y): 1000 for y in range(2021, 2026)}}
MAX_REVIEWS = 5
PLACEBO_PER_CELL = 300

rng = np.random.default_rng(2026)
reviews = pd.read_parquet(DATA / "reviews.parquet")
papers = pd.read_parquet(DATA / "papers.parquet")
ok = papers[~papers.withdrawn & (papers.n_reviews >= 3) & papers.decision.ne("desk_reject")]

chosen = []
for (conf, year), g in ok.groupby(["conf", "year"]):
    n = QUOTA.get((conf, year))
    ids = g.paper_id.to_numpy()
    if n is not None and n < len(ids):
        ids = rng.choice(ids, n, replace=False)
    chosen.append(pd.DataFrame(dict(conf=conf, year=year, paper_id=ids)))
chosen = pd.concat(chosen)

sel = reviews[reviews.paper_id.isin(chosen.paper_id)].copy()
sel["rand"] = rng.random(len(sel))
sel = sel.sort_values("rand").groupby("paper_id").head(MAX_REVIEWS).drop(columns="rand")
sel = sel.sort_values(["conf", "year", "paper_id", "review_id"])
sel[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(DATA / "sample_reviews.parquet")

with open(DATA / "panels.jsonl", "w") as f:
    for pid, g in sel.groupby("paper_id", sort=False):
        f.write(json.dumps(dict(panel_id=pid, review_ids=g.review_id.tolist())) + "\n")

with open(DATA / "panels_placebo.jsonl", "w") as f:
    for (conf, year), g in sel.groupby(["conf", "year"]):
        by_paper = g.groupby("paper_id").review_id.apply(list)
        for k in range(PLACEBO_PER_CELL):
            pids = rng.choice(by_paper.index, 4, replace=False)
            rids = [str(rng.choice(by_paper[p])) for p in pids]
            f.write(json.dumps(dict(panel_id=f"placebo_{conf}_{year}_{k}", review_ids=rids)) + "\n")

print(sel.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())
print("total papers", sel.paper_id.nunique(), "reviews", len(sel), "chars", sel.text.str.len().sum() / 1e6, "M")
