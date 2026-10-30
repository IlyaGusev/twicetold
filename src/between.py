"""Between-paper homogenization: claim templates and critique types, and review-level embedding similarity.

templates: every claim paraphrase is embedded (text-embedding-3-large, 512 dims).
  Template similarity: for each claim, the max cosine to claims of OTHER papers in the same venue-year (fixed reference
  pool per cell). template = above the 75th percentile of ICLR 2017-2022 (a calibrated "template" threshold).
  Critique types: k-means (k=300) on ICLR 2017-2022 claims defines fine-grained types; every claim is assigned to its
  nearest centroid. Tail share = share of claims in types that fall in the bottom half of the 2017-2022 frequency ranking.
  -> data/claims_table_<model>.parquet (claim rows + emb_row, template_sim, template, ctype, tail_type),
     reports/critique_types_<model>.csv, template_by_cell_<model>.csv, type_shift_<model>.csv
embed (Baumann et al. style, next to the claim-level outcomes): every sampled review text is embedded
  (text-embedding-3-small). Per venue-year, IntraSim = mean cosine between reviews of the same paper; InterSim = mean
  cosine between reviews of different papers in the same primary area (300 random pairs per area-year, pooled); plus
  the per-paper IntraSim regressed on the year like S3 in E1. -> reports/embed_<model>.csv

Both read cached embeddings (data/claim_emb_<model>.npy, data/review_emb.npy, row order = claims table and
sample_reviews.parquet). --reembed first rebuilds them with the paid OpenAI API (set -a; . ./.env; set +a).

Usage:
  between.py templates [--model=gpt-6-luna] [--reembed]
  between.py embed [--model=gpt-6-luna] [--reembed] [--threads=4]
  (--data, --reports: where inputs/data outputs and reports live)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
K_TYPES = 300
REF_PER_CELL = 10000  # fixed pool so max-similarity is comparable across cells of different size


def openai_embed(texts, model, batch, threads=4, dtype=np.float32, desc=None, **kw):
    """Embeddings in input order. Retries rate limits; halves texts the API rejects as too long."""
    from openai import APIError, BadRequestError, OpenAI, RateLimitError

    client = OpenAI(max_retries=2)

    def one(chunk):
        for attempt in range(40):
            try:
                r = client.embeddings.create(model=model, input=chunk, **kw)
                return np.array([d.embedding for d in r.data], dtype=dtype)
            except BadRequestError:
                chunk = [t[:len(t) // 2] for t in chunk]
            except (RateLimitError, APIError):
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError("embedding batch failed after retries")

    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(threads) as pool:
        return np.concatenate(list(tqdm(pool.map(one, chunks), total=len(chunks), desc=desc, disable=desc is None)))


def claims_table(model, data):
    reviews = pd.read_parquet(data / "reviews.parquet", columns=["review_id", "paper_id", "conf", "year"]).set_index("review_id")
    rows = []
    with open(data / f"claims_{model}.jsonl") as f:
        for line in tqdm(f, desc="claims table"):
            r = json.loads(line)
            if "claims" not in r or r["review_id"] not in reviews.index:
                continue
            meta = reviews.loc[r["review_id"]]
            for k, c in enumerate(r["claims"]):
                rows.append({"review_id": r["review_id"], "k": k, "paper_id": meta.paper_id, "conf": meta.conf, "year": int(meta.year),
                             "claim": c["claim"], "category": c["category"], "target": c["target"], "specific": c["specific"]})
    return pd.DataFrame(rows)


def embed_claims(model, data):
    T = claims_table(model, data)
    T["emb_row"] = np.arange(len(T))
    T.to_parquet(data / f"claims_table_{model}.parquet")
    E = openai_embed(T.claim.str.slice(0, 400).tolist(), "text-embedding-3-large", batch=512, dtype=np.float16,
                     desc="embed claims", dimensions=512)
    np.save(data / f"claim_emb_{model}.npy", E)


def load_claims(model, data):
    T = pd.read_parquet(data / f"claims_table_{model}.parquet")
    T = T[T.review_id.isin(set(pd.read_parquet(data / "reviews.parquet", columns=["review_id"]).review_id))]
    T = T.reset_index(drop=True)  # drops copied reviews removed from reviews.parquet
    # emb_row keeps the rewritten (filtered) table aligned with the embedding cache; select before the float32 cast
    E = np.load(data / f"claim_emb_{model}.npy", mmap_mode="r")[T.emb_row.to_numpy()].astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return T, E


def template_similarity(T, E, rng):
    pid = T.paper_id.to_numpy()
    out = np.full(len(T), np.nan)
    for idx in tqdm(T.groupby(["conf", "year"]).indices.values(), desc="template sim"):
        ref = rng.choice(idx, min(REF_PER_CELL, len(idx)), replace=False)
        R = E[ref].T
        for s in range(0, len(idx), 4096):
            block = idx[s:s + 4096]
            m = E[block] @ R
            m[pid[block][:, None] == pid[ref][None, :]] = -1
            out[block] = m.max(1)
    return out


def fit_types(E, pre_idx, rng):
    from sklearn.cluster import MiniBatchKMeans

    # fit on a subsample: the full pre-period matrix would need several GB on this machine
    fit_idx = rng.choice(pre_idx, min(300000, len(pre_idx)), replace=False)
    km = MiniBatchKMeans(K_TYPES, random_state=0, batch_size=4096, n_init=3).fit(E[fit_idx])
    return km, np.concatenate([km.predict(E[s:s + 200000]) for s in range(0, len(E), 200000)])


def type_labels(km, pre, E):
    """A readable label per type: the pre-period claim nearest the centroid."""
    centers = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    labels = {}
    for c in range(K_TYPES):
        members = pre.index[pre.ctype == c]
        if len(members):
            labels[c] = pre.claim[members[np.argmax(E[members] @ centers[c])]]
    return pd.Series(labels, name="label")


def cell_summary(T, rng):
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
    return cell


def type_shift(T, labels):
    """Type frequency shifts at ICLR, 2023 -> 2026."""
    I = T[T.conf == "ICLR"]
    f = pd.crosstab(I.ctype, I.year, normalize="columns")
    shift = (f[2026] - f[2023]).sort_values()
    return pd.DataFrame({"label": labels.reindex(shift.index), "f2023": f[2023].reindex(shift.index),
                         "f2026": f[2026].reindex(shift.index), "delta": shift})


def templates(model="gpt-6-luna", reembed=False, data=DATA, reports=REPORTS):
    data, reports = Path(data), Path(reports)
    if reembed:
        embed_claims(model, data)
    T, E = load_claims(model, data)
    rng = np.random.default_rng(0)

    T["template_sim"] = template_similarity(T, E, rng)
    pre = T[(T.conf == "ICLR") & T.year.between(2017, 2022)]
    tau = np.quantile(pre.template_sim, 0.75)
    T["template"] = T.template_sim >= tau
    print(f"template threshold tau={tau:.3f} (75th percentile of ICLR 2017-2022)")

    km, ctype = fit_types(E, pre.index.to_numpy(), rng)
    T["ctype"] = ctype
    pre = pre.assign(ctype=T.ctype[pre.index])
    freq_pre = pre.ctype.value_counts(normalize=True)
    rank = freq_pre.rank(ascending=False)
    tail_types = set(rank[rank > K_TYPES / 2].index)
    T["tail_type"] = T.ctype.isin(tail_types)
    labels = type_labels(km, pre, E)
    labels.to_frame().assign(freq_pre=freq_pre, tail=lambda d: d.index.isin(tail_types)).to_csv(reports / f"critique_types_{model}.csv")
    T.to_parquet(data / f"claims_table_{model}.parquet")

    cell = cell_summary(T, rng)
    cell.round(4).to_csv(reports / f"template_by_cell_{model}.csv")
    print(cell.round(3).to_string())

    out = type_shift(T, labels)
    out.to_csv(reports / f"type_shift_{model}.csv")
    print("\nBiggest falls 2023->2026:\n", out.head(12).round(4).to_string())
    print("\nBiggest rises 2023->2026:\n", out.tail(12).round(4).to_string())


def embed_reviews(data, threads):
    R = pd.read_parquet(data / "sample_reviews.parquet", columns=["review_id", "text"])
    texts = [t[:12000] for t in R.text.fillna("")]  # 8k-token limit; code or LaTeX can reach 1 token per 2 chars
    np.save(data / "review_emb.npy", openai_embed(texts, "text-embedding-3-small", batch=64, threads=threads, desc="embed reviews"))
    print("embedded", len(texts))


def load_reviews(data):
    R = pd.read_parquet(data / "sample_reviews.parquet", columns=["conf", "year", "paper_id", "review_id"])
    meta = pd.read_parquet(data / "reviews.parquet", columns=["review_id", "primary_area"]).drop_duplicates("review_id")
    R = R.merge(meta, on="review_id", how="left", indicator=True)
    E = np.load(data / "review_emb.npy")
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    R["row"] = np.arange(len(R))
    return R[R._merge == "both"].drop(columns="_merge"), E  # drops copied reviews removed from reviews.parquet


def intra_paper(R, E):
    intra = []
    for pid, g in tqdm(R.groupby("paper_id"), desc="intra-paper"):
        if len(g) < 2:
            continue
        M = E[g.row.to_numpy()] @ E[g.row.to_numpy()].T
        iu = np.triu_indices(len(g), 1)
        intra.append({"conf": g.conf.iloc[0], "year": g.year.iloc[0], "paper_id": pid, "intra": float(M[iu].mean())})
    return pd.DataFrame(intra)


def cell_similarity(R, E, I, rng):
    rows = []
    for (conf, year), g in tqdm(R.groupby(["conf", "year"]), desc="inter-paper"):
        sims = []
        for area, h in g.groupby(g.primary_area.fillna("none")):
            if h.paper_id.nunique() < 2:
                continue
            rows_a = h.row.to_numpy()
            for _ in range(300):
                a, b = rng.choice(len(rows_a), 2, replace=False)
                if h.paper_id.iloc[a] != h.paper_id.iloc[b]:
                    sims.append(float(E[rows_a[a]] @ E[rows_a[b]]))
        pooled_rows = g.row.to_numpy()
        inter_any = [float(E[pooled_rows[a]] @ E[pooled_rows[b]]) for a, b in
                     (rng.choice(len(pooled_rows), 2, replace=False) for _ in range(3000))
                     if g.paper_id.iloc[a] != g.paper_id.iloc[b]]
        ii = I[(I.conf == conf) & (I.year == year)].intra
        rows.append({"conf": conf, "year": year, "intra": ii.mean(), "intra_se": ii.std(ddof=1) / np.sqrt(len(ii)),
                     "inter_area": np.mean(sims) if sims else np.nan, "inter_any": np.mean(inter_any), "papers": len(ii)})
    T = pd.DataFrame(rows)
    T["excess"] = T.intra - T.inter_area
    return T


def year_effects(I):
    """ICLR IntraSim year effects relative to 2023, as in E1."""
    import statsmodels.formula.api as smf

    J = I[I.conf == "ICLR"]
    m = smf.ols("intra ~ C(year, Treatment(reference=2023))", data=J).fit(cov_type="HC1")
    print("\nICLR IntraSim year effects vs 2023:")
    for k, v in m.params.items():
        if "year" in k:
            print(f"  {k.split('T.')[1].rstrip(']'):6s} {v:+.4f} (SE {m.bse[k]:.4f})")


def claim_vs_embedding(I, model, data):
    """Paper-level correlation of IntraSim with the claim-level outcomes."""
    P = pd.read_parquet(data / f"panel_outcomes_{model}.parquet", columns=["panel_id", "S3", "share_shared_claims", "D3"])
    P["paper_id"] = P.panel_id.str.replace(r"^(placebo[A]?_|sub3_|noq_|wd_)", "", regex=True)
    Q = I.merge(P, on="paper_id")
    rho = {k: Q[["intra", k]].corr("spearman").iloc[0, 1] for k in ["S3", "share_shared_claims", "D3"]}
    print(f"\npaper-level Spearman: intra vs S3 {rho['S3']:.3f}, vs share shared {rho['share_shared_claims']:.3f}, "
          f"vs D3 {rho['D3']:.3f} (n={len(Q)})")


def embed(model="gpt-6-luna", reembed=False, threads=4, data=DATA, reports=REPORTS):
    data, reports = Path(data), Path(reports)
    if reembed:
        embed_reviews(data, threads)
    R, E = load_reviews(data)
    I = intra_paper(R, E)
    T = cell_similarity(R, E, I, np.random.default_rng(0))
    T.to_csv(reports / f"embed_{model}.csv", index=False)
    print(T.round(4).to_string(index=False))
    year_effects(I)
    claim_vs_embedding(I, model, data)


if __name__ == "__main__":
    fire.Fire({"templates": templates, "embed": embed})
