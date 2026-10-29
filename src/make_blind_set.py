"""Blind reference set: 10 papers each from ICLR 2018, 2022, 2024, 2026 with exactly 4 sampled reviews, full text.

Year is hidden behind codes B00-B39; the key is stored separately. Split into 4 files of 10 papers, each mixing years.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "blind"
YEARS = [2018, 2022, 2024, 2026]
PER_YEAR = 10

rng = np.random.default_rng(42)
sel = pd.read_parquet(DATA / "sample_reviews.parquet")
sel = sel[(sel.conf == "ICLR") & sel.year.isin(YEARS)]
four = sel.groupby("paper_id").filter(lambda g: len(g) == 4)
papers = [p for y in YEARS for p in rng.choice(four[four.year == y].paper_id.unique(), PER_YEAR, replace=False)]
rng.shuffle(papers)

OUT.mkdir(exist_ok=True)
key, items = {}, []
for i, p in enumerate(papers):
    g = four[four.paper_id == p]
    code = f"B{i:02d}"
    key[code] = dict(paper_id=p, year=int(g.year.iloc[0]), review_ids=g.review_id.tolist())
    items.append(dict(paper=code, reviews=[dict(review=f"R{j}", text=t) for j, t in enumerate(g.text)]))
json.dump(key, open(OUT / "blind_key.json", "w"), indent=1)
for k in range(4):
    with open(OUT / f"blind_input_{k}.jsonl", "w") as f:
        for it in items[k::4]:
            f.write(json.dumps(it) + "\n")
with open(DATA / "panels_blind.jsonl", "w") as f:
    for code, v in key.items():
        f.write(json.dumps(dict(panel_id=code, review_ids=v["review_ids"])) + "\n")
print(len(items), "papers;", sum(len(r["text"]) for it in items for r in it["reviews"]), "chars")
