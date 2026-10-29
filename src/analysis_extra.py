"""Extra analyses requested in review: composition decomposition of the hit-rate change, a direct reliability test,
E3 with a claim-count control, category shifts, pooled baseline, raw pre-trend slope, E4 upper bound.
Usage: analysis_extra.py <model>"""

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


def decomposition(model):
    T = pd.read_parquet(DATA / f"pairs_outcomes_{model}.parquet")
    T = T[T.conf == "ICLR"]
    out = []
    for base in [2023, 2024]:
        a, b = T[T.year == base], T[T.year == 2026]
        sa, sb = a.pair_type.value_counts(normalize=True), b.pair_type.value_counts(normalize=True)
        ma, mb = a.groupby("pair_type").hit.mean(), b.groupby("pair_type").hit.mean()
        types = ["HH", "HL", "LL"]
        sa, sb, ma, mb = [x.reindex(types).fillna(0) for x in (sa, sb, ma, mb)]
        total = b.hit.mean() - a.hit.mean()
        comp = float(((sb - sa) * ma).sum())          # shares change, rates at base-year level
        within = float((sb * (mb - ma)).sum())        # rates change, shares at 2026 level
        out.append(dict(base=base, total=total, composition=comp, within_type=within, share_LL_2026=sb["LL"], share_HL_2026=sb["HL"],
                        hit_HH_base=ma["HH"], hit_HH_2026=mb["HH"], hit_LL_2026=mb["LL"]))
    out = pd.DataFrame(out)
    out.to_csv(REPORTS / f"decomposition_{model}.csv", index=False)
    print("Hit-rate change decomposition (ICLR, to 2026):\n", out.round(4).to_string())


def reliability(model):
    """Shared critiques of a flagged-flagged pair: how often does an unflagged co-reviewer of the same paper also raise it?
    Compared with shared critiques of an unflagged-unflagged pair, checked against a third unflagged reviewer."""
    claims = {k: v["claims"] for k, v in load_jsonl(DATA / f"claims_{model}.jsonl", "review_id").items()}
    panels = load_jsonl(DATA / f"clusters_{model}.jsonl", "panel_id")
    meta = pd.read_parquet(DATA / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    flag = pd.read_parquet(DATA / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag
    rows = []
    for pid, p in panels.items():
        if pid not in meta.index or meta.loc[pid, "conf"] != "ICLR" or meta.loc[pid, "year"] < 2024:
            continue
        rids = p["review_ids"]
        fl = [bool(flag.get(r, False)) for r in rids]
        df = assign(p, claims)
        if df.empty:
            continue
        sets = {j: set(df[df.review == j].cluster) for j in range(len(rids))}
        for a, b in combinations(range(len(rids)), 2):
            shared = sets[a] & sets[b]
            if not shared:
                continue
            third = [c for c in range(len(rids)) if c not in (a, b) and not fl[c]]
            if not third:
                continue
            kind = "LL" if fl[a] and fl[b] else "HH" if not fl[a] and not fl[b] else "HL"
            for cl in shared:
                rows.append(dict(paper_id=pid, year=int(meta.loc[pid, "year"]), kind=kind, confirmed=any(cl in sets[c] for c in third), n_third=len(third)))
    R = pd.DataFrame(rows)
    R.to_parquet(DATA / f"reliability_{model}.parquet")
    g = R.groupby("kind").agg(n=("confirmed", "size"), confirmed=("confirmed", "mean"), papers=("paper_id", "nunique"))
    g["se"] = np.sqrt(g.confirmed * (1 - g.confirmed) / g.n)
    g.to_csv(REPORTS / f"reliability_{model}.csv")
    print("\nShare of a pair's shared critiques also raised by an unflagged third reviewer (ICLR 2024-26):\n", g.round(3).to_string())
    # with paper fixed effects: LL vs HH within the same paper
    d = R[R.kind.isin(["HH", "LL"])].copy()
    d["LL"] = (d.kind == "LL").astype(float); d["y"] = d.confirmed.astype(float)
    gg = d.groupby("paper_id")
    for c in ["y", "LL"]:
        d[c + "_dm"] = d[c] - gg[c].transform("mean")
    fit = smf.ols("y_dm ~ 0 + LL_dm", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.paper_id})
    print(f"  within-paper LL - HH: {fit.params['LL_dm']:+.3f} (SE {fit.bse['LL_dm']:.3f}), n={int(fit.nobs)}")


def e3_claim_control(model):
    R = pd.read_parquet(DATA / f"review_outcomes_{model}.parquet")
    R = R[(R.n_claims > 0) & (R.conf == "ICLR") & (R.year >= 2024)].merge(pd.read_parquet(DATA / "detector_scores.parquet", columns=["review_id", "llm_flag"]), on="review_id")
    R["x"] = R.llm_flag.astype(float); R["log_chars"] = np.log(R.chars); R["rating"] = R.rating.fillna(R.rating.median()); R["confidence"] = R.confidence.fillna(R.confidence.median())
    print("\nE3 share_shared on flag, ICLR 2024-26, with claim-count control (paper FE by demeaning):")
    for ctrl in [["log_chars", "rating", "confidence"], ["log_chars", "rating", "confidence", "n_claims"]]:
        d = R.copy(); g = d.groupby("paper_id")
        for c in ["share_shared", "x", *ctrl]:
            d[c] = d[c] - g[c].transform("mean")
        fit = smf.ols("share_shared ~ 0 + x + " + " + ".join(ctrl), data=d).fit(cov_type="cluster", cov_kwds={"groups": d.paper_id})
        print(f"  controls {ctrl}: {fit.params['x']:+.4f} (SE {fit.bse['x']:.4f})")
    # same with only reviews having 4-8 claims (claim-count matched band)
    d = R[R.n_claims.between(4, 8)].copy(); g = d.groupby("paper_id")
    for c in ["share_shared", "x", "log_chars", "rating", "confidence"]:
        d[c] = d[c] - g[c].transform("mean")
    fit = smf.ols("share_shared ~ 0 + x + log_chars + rating + confidence", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.paper_id})
    print(f"  reviews with 4-8 claims only: {fit.params['x']:+.4f} (SE {fit.bse['x']:.4f}), n={int(fit.nobs)}")


def categories(model):
    T = pd.read_parquet(DATA / f"claims_table_{model}.parquet", columns=["review_id", "conf", "year", "category", "specific"])
    T = T[T.conf == "ICLR"]
    flag = pd.read_parquet(DATA / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag
    f = pd.crosstab(T.category, T.year, normalize="columns") * 100
    f["pre21_23"] = f[[2021, 2022, 2023]].mean(axis=1)
    f["d_2026_vs_pre"] = f[2026] - f.pre21_23
    t26 = T[T.year == 2026].copy(); t26["flag"] = t26.review_id.map(flag).fillna(False).astype(bool)
    g = pd.crosstab(t26.category, t26.flag, normalize="columns") * 100
    f["unflagged_2026"], f["flagged_2026"] = g[False], g[True]
    f = f[["pre21_23", 2023, 2024, 2025, 2026, "d_2026_vs_pre", "unflagged_2026", "flagged_2026"]].sort_values("d_2026_vs_pre")
    f.round(2).to_csv(REPORTS / f"category_shift_{model}.csv")
    print("\nCategory shares (% of ICLR claims):\n", f.round(2).to_string())


def pooled_baseline(model):
    P = pd.read_parquet(DATA / f"panel_outcomes_{model}.parquet"); P = P[(P.n_claims > 0) & (P.conf == "ICLR")]
    print("\nICLR 2026 vs pooled 2021-2023 (raw):")
    for y in ["S3", "share_shared_claims", "hit_rate_strict", "D3", "U3"]:
        a, b = P[P.year.between(2021, 2023)][y], P[P.year == 2026][y]
        print(f"  {y:20s} {a.mean():.3f} -> {b.mean():.3f}: {b.mean()-a.mean():+.3f} (SE {np.sqrt(a.var()/len(a)+b.var()/len(b)):.3f})")
    print("\nRaw pre-trend slope per year (ICLR):")
    for yrs in [(2017, 2023), (2018, 2023), (2020, 2023), (2021, 2025)]:
        sub = P[P.year.between(*yrs)]
        fit = smf.ols("S3 ~ year", data=sub).fit(cov_type="HC3")
        print(f"  {yrs}: S3 {fit.params['year']:+.4f} (SE {fit.bse['year']:.4f})")


def e4_bound(model):
    d = pd.read_csv(REPORTS / f"did_{model}.csv")
    r = d[(d.outcome == "S3") & (d.spec == "area+year FE")].iloc[0]
    print(f"\nE4 S3: {r.coef:+.3f} (SE {r.se:.3f}); 95% upper bound {r.coef + 1.96 * r.se:+.3f}; MDE (80% power) {2.8 * r.se:.3f}; "
          f"linear dose implied by the 2026 year effect: {0.275 / 0.445:.2f} per unit")


if __name__ == "__main__":
    m = sys.argv[1]
    decomposition(m); pooled_baseline(m); e4_bound(m); e3_claim_control(m); categories(m); reliability(m)
