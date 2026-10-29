"""More review-requested checks: (1) E4 with the share of both-flagged pairs as the dose and with true 2024-2026 areas;
(2) 2026 unflagged-pair hit rate by the pair's detector-score tercile; (3) critique-type breadth with k-means fitted on
all years, and flagged vs unflagged 2026 claims at equal sample size. Usage: analysis_extra3.py <model>"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import MiniBatchKMeans

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
rng = np.random.default_rng(0)


def e4_variants(model):
    P = pd.read_parquet(DATA / f"panel_outcomes_{model}.parquet")
    P = P[(P.n_claims > 0) & (P.conf == "ICLR")].copy()
    areas = pd.read_parquet(DATA / "paper_areas.parquet")[["paper_id", "area", "area_true"]]
    P = P.drop(columns=["area"], errors="ignore").merge(areas, left_on="panel_id", right_on="paper_id", how="left")
    T = pd.read_parquet(DATA / f"pairs_outcomes_{model}.parquet")
    T = T[T.conf == "ICLR"].merge(areas, on="paper_id", how="left")
    print("E4 variants (S3 on cell dose; area + year FE; SE clustered by cell):")
    for area_col, lab in [("area", "predicted areas"), ("area_true", "true areas, 2023-2026")]:
        d = P.dropna(subset=[area_col]).copy()
        t = T.dropna(subset=[area_col])
        if area_col == "area_true":
            d = d[d.year >= 2023]; t = t[t.year >= 2023]
        cell = t.groupby([area_col, "year"]).agg(dose_LL=("pair_type", lambda s: (s == "LL").mean()), dose_any=("n_flagged", lambda s: (s > 0).mean())).reset_index()
        d = d.merge(cell, on=[area_col, "year"], how="left").dropna(subset=["dose_LL", "S3"])
        for dose in ["dose_LL", "dose_any"]:
            fit = smf.ols(f"S3 ~ {dose} + C({area_col}) + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d[area_col].astype("category").cat.codes * 100 + d.year})
            print(f"  {lab:24s} {dose:9s}: {fit.params[dose]:+.3f} (SE {fit.bse[dose]:.3f}); 95% upper {fit.params[dose] + 1.96 * fit.bse[dose]:+.3f}; n={int(fit.nobs)}; dose range {d[dose].min():.2f}-{d[dose].max():.2f}")


def unflagged_pairs(model):
    T = pd.read_parquet(DATA / f"pairs_outcomes_{model}.parquet")
    det = pd.read_parquet(DATA / "detector_scores.parquet").set_index("review_id")
    T = T[(T.conf == "ICLR") & (T.pair_type == "HH")].copy()
    T["p_pair"] = (det.p_llm.reindex(T.a).to_numpy() + det.p_llm.reindex(T.b).to_numpy()) / 2
    print("\nUnflagged-unflagged pairs: hit rate by year and by tercile of the pair's mean detector score (terciles from 2024-26):")
    cuts = T[T.year >= 2024].p_pair.quantile([1 / 3, 2 / 3]).to_numpy()
    T["terc"] = np.digitize(T.p_pair, cuts)
    print(T[T.year >= 2023].groupby(["year", "terc"]).hit.agg(["mean", "size"]).unstack().round(3).to_string())


def types_robust(model):
    T = pd.read_parquet(DATA / f"claims_table_{model}.parquet", columns=["review_id", "conf", "year", "claim"])
    E = np.load(DATA / f"claim_emb_{model}.npy")
    idx_all = T.index[T.conf == "ICLR"].to_numpy()
    fit_idx = rng.choice(idx_all, 300000, replace=False)
    X = E[fit_idx].astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    km = MiniBatchKMeans(300, random_state=0, batch_size=4096, n_init=3).fit(X)
    flag = pd.read_parquet(DATA / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag

    def hill(idx):
        sub = rng.choice(idx, min(20000, len(idx)), replace=False)
        Z = E[sub].astype(np.float32); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        p = pd.Series(km.predict(Z)).value_counts(normalize=True)
        return float(np.exp(-(p * np.log(p)).sum()))

    print("\nEffective number of critique types with k-means fitted on ALL ICLR years (20k claims per cell):")
    print({y: round(hill(T.index[(T.conf == "ICLR") & (T.year == y)].to_numpy()), 1) for y in range(2017, 2027)})
    t26 = T[(T.conf == "ICLR") & (T.year == 2026)]
    fl = t26.review_id.map(flag).fillna(False).astype(bool)
    print("ICLR 2026, flagged vs unflagged claims (20k each):", round(hill(t26.index[fl].to_numpy()), 1), round(hill(t26.index[~fl].to_numpy()), 1))
    t23 = T[(T.conf == "ICLR") & (T.year == 2023)]
    print("ICLR 2023 (same fit):", round(hill(t23.index.to_numpy()), 1))


if __name__ == "__main__":
    m = sys.argv[1]
    e4_variants(m); unflagged_pairs(m); types_robust(m)
