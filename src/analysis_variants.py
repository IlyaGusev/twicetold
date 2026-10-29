"""Robustness variants: re-clustered 3-review subsets (sub3), questions field dropped (noq), withdrawn 2026 papers (wd).
Usage: analysis_variants.py <model>"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
COLS = ["S3", "D3", "U3", "share_shared_claims", "hit_rate_strict", "pair_rate_strict", "share_specific", "claims_per_review"]


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def main(model):
    P = pd.read_parquet(DATA / f"panel_outcomes_{model}.parquet")
    P = P[P.n_claims > 0].set_index("panel_id")
    out = {}
    # sub3: clustering a 3-review subset from scratch vs rarefied S3 from the full-panel clustering, same papers
    S = pd.read_parquet(DATA / f"sub3_outcomes_{model}.parquet")
    S = S[S.n_claims > 0].set_index("paper_id")
    j = S.join(P[["S3", "D3", "share_shared_claims", "hit_rate_strict"]], rsuffix="_full")
    rows = []
    for y, g in j.groupby("year"):
        rows.append(dict(year=y, n=len(g), S3_recluster=g.S3.mean(), S3_rarefied=g.S3_full.mean(), D3_recluster=g.D3.mean(), D3_rarefied=g.D3_full.mean(),
                         share_recluster=g.share_shared_claims.mean(), share_rarefied=g.share_shared_claims_full.mean(), se_S3=se(g.S3)))
    out["sub3"] = pd.DataFrame(rows).set_index("year")
    # noq: questions field removed, same papers
    N = pd.read_parquet(DATA / f"panel_outcomes_{model}_noq.parquet")
    N = N[N.n_claims > 0].set_index("paper_id")
    j = N.join(P[COLS], rsuffix="_full")
    rows = []
    for y, g in j.groupby("year"):
        rows.append(dict(year=y, n=len(g), **{f"{c}_noq": g[c].mean() for c in ["S3", "share_shared_claims", "claims_per_review", "D3"]},
                         **{f"{c}_full": g[f"{c}_full"].mean() for c in ["S3", "share_shared_claims", "claims_per_review", "D3"]}, se_S3=se(g.S3)))
    out["noq"] = pd.DataFrame(rows).set_index("year")
    # withdrawn ICLR 2026 papers vs the non-withdrawn sample
    W = pd.read_parquet(DATA / f"panel_outcomes_{model}_wd.parquet")
    W = W[W.n_claims > 0]
    nw = P[(P.conf == "ICLR") & (P.year == 2026)]
    out["wd"] = pd.DataFrame({"withdrawn": W[COLS].mean(), "withdrawn_se": W[COLS].agg(se), "kept": nw[COLS].mean(), "kept_se": nw[COLS].agg(se)})
    out["wd"].loc["n"] = [len(W), np.nan, len(nw), np.nan]
    for k, v in out.items():
        v.round(4).to_csv(REPORTS / f"variant_{k}_{model}.csv")
        print(f"== {k}\n{v.round(3).to_string()}\n")


if __name__ == "__main__":
    main(sys.argv[1])
