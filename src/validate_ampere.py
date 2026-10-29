"""Segmentation check against AMPERE (Hua et al. 2019): 400 ICLR 2018 reviews with human proposition labels.

Compares our claim counts with the number of human-labelled evaluation and request propositions, and measures how many
REQUEST propositions (almost always critiques) are covered by at least one extracted claim (cosine >= 0.5, text-embedding-3-small).
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI, RateLimitError
from scipy.stats import spearmanr

DATA = Path(__file__).resolve().parent.parent / "data"
A = pd.read_parquet(DATA / "ampere_full.parquet").set_index("review_id")
claims = {}
for line in open(DATA / "claims_gpt-6-luna_ampere.jsonl"):
    r = json.loads(line)
    if "claims" in r:
        claims[r["review_id"]] = [c["claim"] for c in r["claims"]]
A = A[A.index.isin(claims)]
A["n_claims"] = A.index.map(lambda i: len(claims[i]))
print(len(A), "reviews")
for k in ["n_prop", "n_eval", "n_req"]:
    print(f"spearman(n_claims, {k}) = {spearmanr(A.n_claims, A[k]).statistic:.3f}")
A["n_crit"] = A.n_eval + A.n_req
print(f"spearman(n_claims, n_eval+n_req) = {spearmanr(A.n_claims, A.n_crit).statistic:.3f}; means claims {A.n_claims.mean():.1f}, eval+req {A.n_crit.mean():.1f}, req {A.n_req.mean():.1f}")

client = OpenAI(max_retries=2)


def embed(texts):
    out = []
    for i in range(0, len(texts), 512):
        while True:
            try:
                r = client.embeddings.create(model="text-embedding-3-small", input=texts[i:i + 512])
                out += [e.embedding for e in r.data]
                break
            except RateLimitError:
                time.sleep(10)
    e = np.array(out, dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


cov, cov_eval, prec = [], [], []
for rid, row in A.iterrows():
    cl = claims[rid]
    if not cl:
        cov += [0] * row.n_req
        continue
    sents = list(row.sents)
    E = embed(sents + cl)
    S, C = E[:len(sents)], E[len(sents):]
    sim = S @ C.T
    for s, lab in zip(sim.max(1), row.labels):
        if lab == "request":
            cov.append(s >= 0.5)
        if lab == "evaluation":
            cov_eval.append(s >= 0.5)
    arg = [l in ("evaluation", "request") for l in row.labels]
    best = sim.argmax(0)
    prec += [arg[b] for b in best]
print(f"request propositions covered by a claim: {np.mean(cov):.3f} (n={len(cov)})")
print(f"evaluation propositions covered by a claim: {np.mean(cov_eval):.3f} (n={len(cov_eval)})")
print(f"claims whose nearest proposition is evaluation/request: {np.mean(prec):.3f} (n={len(prec)})")
