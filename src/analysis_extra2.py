"""Reliability test with one third reviewer at a time; overlap on substantive claims only (no clarity/typo claims);
sub-threshold detector score among unflagged reviews. Usage: analysis_extra2.py <model>"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from metrics import assign, load_jsonl, rarefy

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
NONSUBST = {"clarity/presentation", "typo/formatting"}


def main(model):
    claims = {k: v["claims"] for k, v in load_jsonl(DATA / f"claims_{model}.jsonl", "review_id").items()}
    panels = load_jsonl(DATA / f"clusters_{model}.jsonl", "panel_id")
    meta = pd.read_parquet(DATA / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    flag = pd.read_parquet(DATA / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag
    rel, subst = [], []
    for pid, p in panels.items():
        if pid not in meta.index or meta.loc[pid, "conf"] != "ICLR":
            continue
        year = int(meta.loc[pid, "year"])
        df = assign(p, claims)
        if df.empty:
            continue
        s = df[~df.category.isin(NONSUBST)]
        if len(s) and s.review.nunique() >= 3:
            (d3, s3, u3), = rarefy(s, "cluster", (3,)).values()
            reviewers = s.groupby("cluster").review.nunique()
            subst.append(dict(paper_id=pid, year=year, S3=s3, D3=d3, share=float(s.cluster.map(reviewers).ge(2).mean()), n=len(s)))
        if year < 2024:
            continue
        rids = p["review_ids"]
        fl = [bool(flag.get(r, False)) for r in rids]
        sets = {j: set(df[df.review == j].cluster) for j in range(len(rids))}
        for a, b in combinations(range(len(rids)), 2):
            shared = sets[a] & sets[b]
            kind = "LL" if fl[a] and fl[b] else "HH" if not fl[a] and not fl[b] else "HL"
            for c in range(len(rids)):
                if c in (a, b) or fl[c]:
                    continue
                for cl in shared:
                    rel.append(dict(paper_id=pid, kind=kind, confirmed=cl in sets[c]))
    R = pd.DataFrame(rel)
    g = R.groupby("kind").agg(n=("confirmed", "size"), confirmed=("confirmed", "mean"), papers=("paper_id", "nunique"))
    g["se"] = np.sqrt(g.confirmed * (1 - g.confirmed) / g.n)
    g.to_csv(REPORTS / f"reliability_{model}.csv")
    print("Share of a pair's shared critiques that one given unflagged third reviewer also raises (ICLR 2024-26):\n", g.round(3).to_string())
    d = R[R.kind.isin(["HH", "LL"])].copy()
    d["LL"] = (d.kind == "LL").astype(float); d["y"] = d.confirmed.astype(float)
    gg = d.groupby("paper_id")
    for c in ["y", "LL"]:
        d[c + "_dm"] = d[c] - gg[c].transform("mean")
    fit = smf.ols("y_dm ~ 0 + LL_dm", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.paper_id})
    print(f"  within-paper LL - HH: {fit.params['LL_dm']:+.3f} (SE {fit.bse['LL_dm']:.3f}), n={int(fit.nobs)}")

    S = pd.DataFrame(subst)
    S.to_parquet(DATA / f"substantive_{model}.parquet")
    m = S.groupby("year")[["S3", "D3", "share"]].agg(["mean", "sem"]).round(3)
    m.to_csv(REPORTS / f"substantive_{model}.csv")
    print("\nOverlap on substantive claims only (clarity/typo claims removed), ICLR:\n", m.to_string())

    # sub-threshold style among unflagged reviews: within-paper association of p_llm with share_shared, 2026
    Rv = pd.read_parquet(DATA / f"review_outcomes_{model}.parquet")
    det = pd.read_parquet(DATA / "detector_scores.parquet")
    Rv = Rv[(Rv.n_claims > 0) & (Rv.conf == "ICLR")].merge(det, on="review_id")
    print("\nAmong UNFLAGGED reviews, within-paper association of the continuous detector score with share_shared:")
    for yrs in [(2024, 2026), (2026, 2026), (2019, 2022)]:
        d = Rv[~Rv.llm_flag & Rv.year.between(*yrs)].copy()
        d["x"] = d.p_llm; d["log_chars"] = np.log(d.chars); d["rating"] = d.rating.fillna(d.rating.median()); d["confidence"] = d.confidence.fillna(d.confidence.median())
        g = d.groupby("paper_id")
        for c in ["share_shared", "x", "log_chars", "rating", "confidence"]:
            d[c] = d[c] - g[c].transform("mean")
        fit = smf.ols("share_shared ~ 0 + x + log_chars + rating + confidence", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.paper_id})
        print(f"  {yrs}: per unit p_llm {fit.params['x']:+.4f} (SE {fit.bse['x']:.4f}); per SD of p_llm among unflagged {fit.params['x']*Rv[~Rv.llm_flag & Rv.year.between(*yrs)].p_llm.std():+.4f}; n={int(fit.nobs)}")
    print("mean p_llm among unflagged by year:", Rv[~Rv.llm_flag].groupby("year").p_llm.mean().round(4).to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
