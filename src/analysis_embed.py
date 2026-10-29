"""Embedding-level homogenization (Baumann et al. style) next to the claim-level outcomes.

Usage: analysis_embed.py embed|analyze
embed: text-embedding-3-small of every sampled review text -> data/review_emb.npy (row order = sample_reviews.parquet).
analyze: per venue-year, IntraSim = mean cosine between reviews of the same paper; InterSim = mean cosine between
reviews of different papers in the same primary area (300 random pairs per area-year, pooled); plus the per-paper
IntraSim regressed on the year like S3 in E1. -> reports/embed_gpt-6-luna.csv
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"


def embed():
    from openai import APIError, BadRequestError, OpenAI, RateLimitError

    client = OpenAI()
    R = pd.read_parquet(DATA / "sample_reviews.parquet", columns=["review_id", "text"])
    texts = [t[:12000] for t in R.text.fillna("")]  # 8k-token limit; code or LaTeX can reach 1 token per 2 chars
    out = np.zeros((len(texts), 1536), dtype=np.float32)

    def run(i):
        batch = texts[i:i + 64]
        for attempt in range(40):
            try:
                r = client.embeddings.create(model="text-embedding-3-small", input=batch)
                break
            except BadRequestError:
                batch = [t[:len(t) // 2] for t in batch]
            except (RateLimitError, APIError):
                time.sleep(min(30, 2 ** attempt))
        else:
            raise RuntimeError(f"batch {i} failed after retries")
        out[i:i + len(batch)] = np.array([d.embedding for d in r.data], dtype=np.float32)

    with ThreadPoolExecutor(int(os.environ.get("THREADS", 4))) as ex:
        list(ex.map(run, range(0, len(texts), 64)))
    np.save(DATA / "review_emb.npy", out)
    print("embedded", len(texts))


def analyze():
    import statsmodels.formula.api as smf

    R = pd.read_parquet(DATA / "sample_reviews.parquet", columns=["conf", "year", "paper_id", "review_id"])
    meta = pd.read_parquet(DATA / "reviews.parquet", columns=["review_id", "primary_area"]).drop_duplicates("review_id")
    R = R.merge(meta, on="review_id", how="left")
    E = np.load(DATA / "review_emb.npy")
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    R["row"] = np.arange(len(R))
    rng = np.random.default_rng(0)
    intra = []
    for pid, g in R.groupby("paper_id"):
        if len(g) < 2:
            continue
        M = E[g.row.to_numpy()] @ E[g.row.to_numpy()].T
        iu = np.triu_indices(len(g), 1)
        intra.append(dict(conf=g.conf.iloc[0], year=g.year.iloc[0], paper_id=pid, intra=float(M[iu].mean())))
    I = pd.DataFrame(intra)
    rows = []
    for (conf, year), g in R.groupby(["conf", "year"]):
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
        rows.append(dict(conf=conf, year=year, intra=ii.mean(), intra_se=ii.std(ddof=1) / np.sqrt(len(ii)),
                         inter_area=np.mean(sims) if sims else np.nan, inter_any=np.mean(inter_any), papers=len(ii)))
    T = pd.DataFrame(rows)
    T["excess"] = T.intra - T.inter_area
    T.to_csv(REPORTS / "embed_gpt-6-luna.csv", index=False)
    print(T.round(4).to_string(index=False))
    # year effects relative to 2023, ICLR, as in E1
    J = I[I.conf == "ICLR"].copy()
    J["y"] = pd.Categorical(J.year.astype(str), categories=sorted(J.year.astype(str).unique()))
    m = smf.ols("intra ~ C(year, Treatment(reference=2023))", data=J).fit(cov_type="HC1")
    print("\nICLR IntraSim year effects vs 2023:")
    for k, v in m.params.items():
        if "year" in k:
            print(f"  {k.split('T.')[1].rstrip(']'):6s} {v:+.4f} (SE {m.bse[k]:.4f})")
    # claim-level vs embedding-level, paper level correlation
    P = pd.read_parquet(DATA / "panel_outcomes_gpt-6-luna.parquet", columns=["panel_id", "S3", "share_shared_claims", "D3"])
    P["paper_id"] = P.panel_id.str.replace(r"^(placebo[A]?_|sub3_|noq_|wd_)", "", regex=True)
    Q = I.merge(P, on="paper_id")
    print("\npaper-level Spearman: intra vs S3 %.3f, vs share shared %.3f, vs D3 %.3f (n=%d)" % (
        Q[["intra", "S3"]].corr("spearman").iloc[0, 1], Q[["intra", "share_shared_claims"]].corr("spearman").iloc[0, 1],
        Q[["intra", "D3"]].corr("spearman").iloc[0, 1], len(Q)))


if __name__ == "__main__":
    {"embed": embed, "analyze": analyze}[sys.argv[1]]()
