"""E3 at the review-pair level: within the same paper, do human-human, human-LLM and LLM-LLM review pairs differ in
how often their claims match? Pair hit rate = mean share of one review's claims matched in the other (strict clusters).

Usage: analysis_pairs.py <model> [tag]
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from metrics import assign, load_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def pair_table(model, tag):
    claims = {k: v["claims"] for k, v in load_jsonl(DATA / f"claims_{model}{tag}.jsonl", "review_id").items()}
    panels = load_jsonl(DATA / f"clusters_{model}{tag}.jsonl", "panel_id")
    meta = pd.read_parquet(DATA / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    det = pd.read_parquet(DATA / "detector_scores.parquet").set_index("review_id")
    rows = []
    for pid, p in panels.items():
        if pid not in meta.index:
            continue
        df = assign(p, claims)
        if df.empty:
            continue
        for a, b in combinations(sorted(df.review.unique()), 2):
            ga, gb = df[df.review == a], df[df.review == b]
            ra, rb = p["review_ids"][a], p["review_ids"][b]
            if ra not in det.index or rb not in det.index:
                continue
            for suffix, xa, xb in [("", ga, gb), ("_spec", ga[ga.specific.astype(bool)], gb[gb.specific.astype(bool)])]:
                if len(xa) == 0 or len(xb) == 0:
                    rows.append(dict(paper_id=pid, a=ra, b=rb, **{f"hit{suffix}": np.nan, f"pair{suffix}": np.nan}))
                    continue
                same = xa.cluster.to_numpy()[:, None] == xb.cluster.to_numpy()[None, :]
                rows.append(dict(paper_id=pid, a=ra, b=rb, **{f"hit{suffix}": (same.any(1).mean() + same.any(0).mean()) / 2,
                                                            f"pair{suffix}": same.mean()}))
    T = pd.DataFrame(rows).groupby(["paper_id", "a", "b"], as_index=False).first()
    T["n_flagged"] = det.llm_flag.reindex(T.a).to_numpy().astype(int) + det.llm_flag.reindex(T.b).to_numpy().astype(int)
    T["pair_type"] = T.n_flagged.map({0: "HH", 1: "HL", 2: "LL"})
    T = T.join(meta, on="paper_id")
    return T


def main(model, tag=""):
    T = pd.read_parquet(DATA / f"pairs_outcomes_{model}{tag}.parquet") if (DATA / f"pairs_outcomes_{model}{tag}.parquet").exists() else pair_table(model, tag)
    T.to_parquet(DATA / f"pairs_outcomes_{model}{tag}.parquet")
    rows = []
    for conf, years in [("ICLR", [2024, 2025, 2026]), ("ICLR", [2026]), ("NeurIPS", [2023, 2024, 2025])]:
        d = T[(T.conf == conf) & T.year.isin(years)].copy()
        d["HL"] = (d.pair_type == "HL").astype(float)
        d["LL"] = (d.pair_type == "LL").astype(float)
        for y in ["hit", "pair", "hit_spec", "pair_spec"]:
            dd = d.dropna(subset=[y]).copy()
            if (dd.pair_type == "HH").sum() == 0 or dd.pair_type.nunique() < 2:
                continue
            # paper fixed effects by within-paper demeaning (dummies for thousands of papers do not fit in memory)
            g = dd.groupby("paper_id")
            for c in [y, "HL", "LL"]:
                dd[c + "_dm"] = dd[c] - g[c].transform("mean")
            fit = smf.ols(f"{y}_dm ~ 0 + HL_dm + LL_dm", data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd.paper_id})
            for t in ["HL", "LL"]:
                k = f"{t}_dm"
                rows.append(dict(conf=conf, years=f"{years[0]}-{years[-1]}", outcome=y, pair_type=t, coef=fit.params[k], se=fit.bse[k],
                                 p=fit.pvalues[k], n=int(fit.nobs), mean_HH=dd[dd.pair_type == "HH"][y].mean(),
                                 n_HH=int((dd.pair_type == "HH").sum()), n_t=int((dd.pair_type == t).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / f"pairs_within_{model}{tag}.csv", index=False)
    print("Pair-level within-paper effects vs human-human pairs (paper FE by demeaning, clustered SE):")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
