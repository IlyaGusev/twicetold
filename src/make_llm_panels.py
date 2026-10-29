"""E5 panels from the IntelLabs benchmark: for ICLR 2022-2024 papers in our sample, all-LLM panels (4 different LLMs)
and mixed panels (2 human + 2 LLM reviews). Writes data/llm_reviews.parquet (pass-1 input) and data/panels_llm.jsonl."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
PER_YEAR = 500
rng = np.random.default_rng(5)
I = pd.read_parquet(DATA / "intel_reviews.parquet")
I = I[I.paper_id.notna() & I.year.isin([2022, 2023, 2024])]
panels = {json.loads(l)["panel_id"]: json.loads(l)["review_ids"] for l in open(DATA / "panels.jsonl")}
llm = I[I.label == 1]
by_paper = llm.groupby("paper_id")
ok = [p for p, g in by_paper if p in panels and len(panels[p]) >= 4 and g.llm.nunique() >= 4]
out, reviews = [], set()
for year in (2022, 2023, 2024):
    cand = [p for p in ok if llm[llm.paper_id == p].year.iloc[0] == year]
    chosen = rng.choice(cand, min(PER_YEAR, len(cand)), replace=False)
    for p in chosen:
        g = llm[llm.paper_id == p].drop_duplicates("llm")
        models = ["gpt4o", "claude", "gemini", rng.choice(["llama", "qwen"])]
        g = g[g.llm.isin(models)]
        if len(g) < 4:
            continue
        lids = g.review_id.tolist()
        hids = list(rng.choice(panels[p], 4, replace=False))
        out.append(dict(panel_id=f"llm4_{p}", review_ids=lids))
        out.append(dict(panel_id=f"mix22_{p}", review_ids=hids[:2] + list(rng.choice(lids, 2, replace=False))))
        out.append(dict(panel_id=f"hum4_{p}", review_ids=hids))
        reviews.update(lids)
with open(DATA / "panels_llm.jsonl", "w") as f:
    for p in out:
        f.write(json.dumps(p) + "\n")
llm[llm.review_id.isin(reviews)][["review_id", "text"]].to_parquet(DATA / "llm_reviews.parquet")
print(len(out), "panels;", len(reviews), "LLM reviews")
