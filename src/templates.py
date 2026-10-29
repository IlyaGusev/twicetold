"""Cross-paper analysis of claims: template similarity, fine-grained critique types, and tail loss.

Steps:
1. Embed every claim paraphrase (text-embedding-3-large) -> data/claim_emb_<model>.npy and data/claims_table_<model>.parquet.
2. Template similarity: for each claim, the max cosine to claims of OTHER papers in the same venue-year (20k reference
   claims per cell). G = share of claims above the 75th percentile of ICLR 2017-2022 (a calibrated "template" threshold).
3. Critique types: k-means (k=300) on ICLR 2017-2022 claims defines fine-grained types; every claim is assigned to its
   nearest centroid. Tail share = share of claims in types that fall in the bottom half of the 2017-2022 frequency ranking.

Usage: templates.py <model> embed|analyze
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI, RateLimitError
from sklearn.cluster import MiniBatchKMeans

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
K_TYPES = 300
REF_PER_CELL = 10000  # fixed pool so max-similarity is comparable across cells of different size
rng = np.random.default_rng(0)


def claims_table(model):
    reviews = pd.read_parquet(DATA / "reviews.parquet", columns=["review_id", "paper_id", "conf", "year"]).set_index("review_id")
    rows = []
    for line in open(DATA / f"claims_{model}.jsonl"):
        r = json.loads(line)
        if "claims" not in r or r["review_id"] not in reviews.index:
            continue
        meta = reviews.loc[r["review_id"]]
        for k, c in enumerate(r["claims"]):
            rows.append(dict(review_id=r["review_id"], k=k, paper_id=meta.paper_id, conf=meta.conf, year=int(meta.year),
                             claim=c["claim"], category=c["category"], target=c["target"], specific=c["specific"]))
    return pd.DataFrame(rows)


def embed_all(model):
    T = claims_table(model)
    T.to_parquet(DATA / f"claims_table_{model}.parquet")
    client = OpenAI(max_retries=2)
    texts = T.claim.str.slice(0, 400).tolist()
    chunks = [texts[i:i + 512] for i in range(0, len(texts), 512)]
    print(len(texts), "claims,", len(chunks), "chunks")

    def one(chunk):
        while True:
            try:
                r = client.embeddings.create(model="text-embedding-3-large", input=chunk, dimensions=512)
                return np.array([e.embedding for e in r.data], dtype=np.float16)
            except RateLimitError:
                time.sleep(10)

    with ThreadPoolExecutor(4) as pool:
        emb = np.concatenate(list(pool.map(one, chunks)))
    np.save(DATA / f"claim_emb_{model}.npy", emb)


def analyze(model):
    T = pd.read_parquet(DATA / f"claims_table_{model}.parquet")
    E = np.load(DATA / f"claim_emb_{model}.npy").astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    pid = T.paper_id.to_numpy()

    # 2. template similarity within venue-year
    T["template_sim"] = np.nan
    for (conf, year), idx in T.groupby(["conf", "year"]).indices.items():
        ref = rng.choice(idx, min(REF_PER_CELL, len(idx)), replace=False)
        sims = np.empty(len(idx), dtype=np.float32)
        for s in range(0, len(idx), 4096):
            block = idx[s:s + 4096]
            m = E[block] @ E[ref].T
            m[pid[block][:, None] == pid[ref][None, :]] = -1
            sims[s:s + 4096] = m.max(1)
        T.loc[T.index[idx], "template_sim"] = sims
    pre = T[(T.conf == "ICLR") & T.year.between(2017, 2022)]
    tau = np.quantile(pre.template_sim, 0.75)
    T["template"] = T.template_sim >= tau
    print(f"template threshold tau={tau:.3f} (75th percentile of ICLR 2017-2022)")

    # 3. fine-grained types from the pre-period
    # fit on a subsample: the full pre-period matrix would need several GB on this machine
    fit_idx = rng.choice(pre.index.to_numpy(), min(300000, len(pre)), replace=False)
    km = MiniBatchKMeans(K_TYPES, random_state=0, batch_size=4096, n_init=3).fit(E[fit_idx])
    T["ctype"] = np.concatenate([km.predict(E[s:s + 200000]) for s in range(0, len(E), 200000)])
    pre = pre.assign(ctype=T.ctype[pre.index])
    freq_pre = pre.ctype.value_counts(normalize=True)
    rank = freq_pre.rank(ascending=False)
    tail_types = set(rank[rank > K_TYPES / 2].index)
    T["tail_type"] = T.ctype.isin(tail_types)
    # a readable label per type: the claim nearest the centroid
    centers = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    labels = {}
    for c in range(K_TYPES):
        members = pre.index[pre.ctype == c]
        if len(members):
            labels[c] = pre.claim[members[np.argmax(E[members] @ centers[c])]]
    pd.Series(labels, name="label").to_frame().assign(freq_pre=freq_pre, tail=lambda d: d.index.isin(tail_types)).to_csv(REPORTS / f"critique_types_{model}.csv")
    T.to_parquet(DATA / f"claims_table_{model}.parquet")

    cell = T.groupby(["conf", "year"]).agg(claims=("claim", "size"), template_sim=("template_sim", "mean"),
                                           template_share=("template", "mean"), tail_share=("tail_type", "mean"),
                                           specific=("specific", "mean"))
    # effective number of types (Hill-Shannon) per cell, rarefied to 20k claims
    hill = {}
    for key, idx in T.groupby(["conf", "year"]).indices.items():
        sub = rng.choice(idx, min(20000, len(idx)), replace=False)
        p = T.ctype.iloc[sub].value_counts(normalize=True)
        hill[key] = float(np.exp(-(p * np.log(p)).sum()))
    cell["hill_types_20k"] = pd.Series(hill)
    cell.round(4).to_csv(REPORTS / f"template_by_cell_{model}.csv")
    print(cell.round(3).to_string())

    # type frequency shifts: top growers and shrinkers ICLR 2023 -> 2026
    f = pd.crosstab(T[T.conf == "ICLR"].ctype, T[T.conf == "ICLR"].year, normalize="columns")
    shift = (f[2026] - f[2023]).sort_values()
    lab = pd.Series(labels)
    out = pd.DataFrame(dict(label=lab.reindex(shift.index), f2023=f[2023].reindex(shift.index), f2026=f[2026].reindex(shift.index), delta=shift))
    out.to_csv(REPORTS / f"type_shift_{model}.csv")
    print("\nBiggest falls 2023->2026:\n", out.head(12).round(4).to_string())
    print("\nBiggest rises 2023->2026:\n", out.tail(12).round(4).to_string())


if __name__ == "__main__":
    {"embed": embed_all, "analyze": analyze}[sys.argv[2]](sys.argv[1])
