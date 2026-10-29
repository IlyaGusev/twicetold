"""E3: within-paper association between review-level LLM exposure and claim overlap, with paper fixed effects.

Usage: analysis_within.py <model> [tag]
Exposure columns are taken from data/review_outcomes_<model><tag>.parquet (marker words) and, if present,
data/detector_scores.parquet (style detector probability and grade). Clustered SEs by paper.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
OUTCOMES = ["share_shared", "share_shared_lenient", "n_claims", "n_unique", "share_specific",
            "cat_clarity/presentation", "cat_typo/formatting", "cat_ablation", "cat_generalization", "cat_related-work",
            "cat_statistical-rigor", "cat_experiments-scope", "cat_compute-cost/scalability"]


def fit_within(R, outcome, exposure, controls=("log_chars", "rating", "confidence")):
    """Paper fixed effects by within-paper demeaning (30k paper dummies do not fit in memory)."""
    d = R.dropna(subset=[outcome, exposure]).copy()
    d["y"] = d[outcome]
    d["x"] = d[exposure].astype(float)
    d["log_chars"] = np.log(d.chars)
    cols = ["y", "x", *controls]
    g = d.groupby("paper_id")
    for c in cols:
        d[c] = d[c] - g[c].transform("mean")
    fit = smf.ols("y ~ 0 + x + " + " + ".join(controls), data=d).fit(cov_type="cluster", cov_kwds={"groups": d.paper_id})
    return fit.params["x"], fit.bse["x"], fit.pvalues["x"], len(d)


def spillover(R, model, tag):
    """Outcomes of UNFLAGGED reviews as a function of how many co-reviews of the same paper are flagged.
    Paper FE are not possible (own + others' flags sum to a paper constant), so this uses year FE and controls."""
    if "llm_flag" not in R:
        return
    d = R[R.conf == "ICLR"].copy()
    d["llm_flag"] = d.llm_flag.fillna(False).astype(int)
    tot = d.groupby("paper_id").llm_flag.transform("sum")
    d["n_flagged_others"] = tot - d.llm_flag
    d["n_others"] = d.groupby("paper_id").review_id.transform("size") - 1
    d = d[(d.llm_flag == 0) & (d.year >= 2024)]
    rows = []
    for y in ["share_shared", "n_claims", "n_unique", "share_specific"]:
        f = f"{y} ~ n_flagged_others + n_others + np.log(chars) + rating + confidence + C(year)"
        fit = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.paper_id})
        rows.append(dict(outcome=y, coef=fit.params["n_flagged_others"], se=fit.bse["n_flagged_others"], p=fit.pvalues["n_flagged_others"], n=int(fit.nobs)))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / f"spillover_{model}{tag}.csv", index=False)
    print("\nSpillover: unflagged reviews, ICLR 2024-26, per additional flagged co-review")
    print(out.round(4).to_string())


def main(model, tag=""):
    R = pd.read_parquet(DATA / f"review_outcomes_{model}{tag}.parquet")
    R = R[R.n_claims > 0].copy()
    R["rating"] = R.rating.fillna(R.rating.median())
    R["confidence"] = R.confidence.fillna(R.confidence.median())
    for c in OUTCOMES:
        if c.startswith("cat_"):
            R[c.replace("/", "_").replace("-", "_")] = R[c]
    det = DATA / "detector_scores.parquet"
    exposures = ["any_marker", "marker_rate"]
    if det.exists():
        R = R.merge(pd.read_parquet(det), on="review_id", how="left")
        exposures += [c for c in ["p_llm", "llm_grade", "llm_flag", "p_llm_ss", "llm_flag_ss"] if c in R]
    gz = DATA / "gptzero.parquet"
    if gz.exists():
        R = R.merge(pd.read_parquet(gz, columns=["review_id", "gz_ai", "gz_isAI"]), on="review_id", how="left")
        exposures += ["gz_isAI", "gz_ai"]
    rows = []
    for conf, years in [("ICLR", [2024, 2025, 2026]), ("ICLR", [2026]), ("ICLR", [2024]), ("ICLR", [2025]), ("NeurIPS", [2023, 2024, 2025]),
                        ("ICLR", [2019, 2020, 2021, 2022])]:
        sub = R[(R.conf == conf) & R.year.isin(years)]
        for e in exposures:
            if e not in sub or sub[e].isna().all():
                continue
            for y in OUTCOMES:
                yc = y.replace("/", "_").replace("-", "_")
                try:
                    b, se, p, n = fit_within(sub, yc, e)
                except Exception as ex:  # noqa: BLE001
                    print("skip", conf, years, e, y, ex)
                    continue
                rows.append(dict(conf=conf, years="-".join(map(str, [years[0], years[-1]])), exposure=e, outcome=y,
                                 coef=b, se=se, p=p, n=n, mean_y=sub[yc].mean(), sd_y=sub[yc].std()))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / f"within_paper_{model}{tag}.csv", index=False)
    spillover(R, model, tag)
    for (conf, yrs, e), g in out.groupby(["conf", "years", "exposure"]):
        print(f"\n== {conf} {yrs}, exposure={e}, n={g.n.iloc[0]}")
        for _, r in g.iterrows():
            print(f"  {r.outcome:28s} {r.coef:+.4f} (SE {r.se:.4f}) p={r.p:.3f}  mean={r.mean_y:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
