"""Primary-area labels for every sampled paper.

ICLR 2026 submissions carry an OpenReview primary_area. We embed title + abstract (text-embedding-3-small), fit a
logistic regression on the 2026 labels, and predict the area of earlier papers. Output: data/paper_areas.parquet with
columns paper_id, area, area_prob, embedding row index into data/abstract_emb.npy.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import time

from openai import OpenAI, RateLimitError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

DATA = Path(__file__).resolve().parent.parent / "data"
papers = pd.read_parquet(DATA / "papers.parquet")
sampled = set(json.loads(l)["panel_id"] for l in open(DATA / "panels.jsonl"))
need = papers[papers.paper_id.isin(sampled) | ((papers.year == 2026) & papers.primary_area.notna())].copy()
need["doc"] = (need.title.fillna("") + "\n" + need.abstract.fillna("")).str.slice(0, 6000)
need = need[need.doc.str.len() > 50].reset_index(drop=True)
print(len(need), "papers to embed")

client = OpenAI(max_retries=2)


def embed(chunk):
    while True:
        try:
            r = client.embeddings.create(model="text-embedding-3-small", input=chunk)
            return [e.embedding for e in r.data]
        except RateLimitError:
            time.sleep(10)


docs = need.doc.tolist()
chunks = [docs[i:i + 256] for i in range(0, len(docs), 256)]
with ThreadPoolExecutor(2) as pool:
    emb = np.array([e for part in pool.map(embed, chunks) for e in part], dtype=np.float32)
np.save(DATA / "abstract_emb.npy", emb)

labelled = need.primary_area.notna().to_numpy()
y = need.primary_area[labelled].to_numpy()
clf = LogisticRegression(max_iter=2000, C=3.0)
print("5-fold accuracy on 2026 labels:", cross_val_score(clf, emb[labelled], y, cv=5).mean().round(3))
clf.fit(emb[labelled], y)
proba = clf.predict_proba(emb)
pred = clf.classes_[proba.argmax(1)]
out = pd.DataFrame(dict(paper_id=need.paper_id, area=np.where(labelled, need.primary_area, pred),
                        area_prob=proba.max(1), area_pred=pred, row=np.arange(len(need))))
out.to_parquet(DATA / "paper_areas.parquet")
print(out.merge(papers[["paper_id", "conf", "year"]]).groupby(["conf", "year"]).area.value_counts(normalize=True)
      .unstack(fill_value=0).round(2).T.to_string())
