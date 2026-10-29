"""E4: area-by-year dose design. Cell-level LLM exposure (share of reviews flagged by the style detector, or GPTZero for
2018-2024) as a continuous treatment, with area and year fixed effects and area-specific linear trends.

Usage: analysis_did.py <model> [tag]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
OUTCOMES = ["S3", "U3", "D3", "share_shared_claims", "pair_rate_strict", "hit_rate_strict", "share_specific", "claims_per_review"]


def main(model, tag=""):
    P = pd.read_parquet(DATA / f"panel_outcomes_{model}{tag}.parquet")
    P = P[(P.n_claims > 0) & (P.conf == "ICLR")].copy()
    areas = pd.read_parquet(DATA / "paper_areas.parquet")[["paper_id", "area", "area_final"]]
    P = P.drop(columns=["area"], errors="ignore").merge(areas, left_on="panel_id", right_on="paper_id", how="left")
    R = pd.read_parquet(DATA / f"review_outcomes_{model}{tag}.parquet", columns=["review_id", "paper_id", "year"])
    det = pd.read_parquet(DATA / "detector_scores.parquet")
    R = R.merge(det, on="review_id", how="left").merge(areas, on="paper_id", how="left")
    cell = R.groupby(["area", "year"]).agg(exposure=("llm_flag", "mean"), p_llm=("p_llm", "mean"), n=("review_id", "size")).reset_index()
    P = P.merge(cell, on=["area", "year"], how="left")
    P["t"] = P.year - 2017
    P["post"] = (P.year >= 2024).astype(int)
    P["confidence_mean"] = P.confidence_mean.fillna(P.confidence_mean.median())
    P["rating_sd"] = P.rating_sd.fillna(P.rating_sd.median())
    P["rating_mean"] = P.rating_mean.fillna(P.rating_mean.median())
    # ratings are standardized within venue-year because the ICLR 2026 scale changed
    g = P.groupby(["conf", "year"])
    P["rating_z"] = (P.rating_mean - g.rating_mean.transform("mean")) / g.rating_mean.transform("std")
    P["rating_sd_z"] = P.rating_sd / g.rating_sd.transform("mean")
    print("cell exposure (share flagged) by area x year, ICLR:")
    print(cell.pivot(index="area", columns="year", values="exposure").round(3).to_string())
    rows = []
    for y in OUTCOMES:
        for spec, f in [("area+year FE", f"{y} ~ exposure + C(area) + C(year) + np.log(total_chars) + rating_z + rating_sd_z + confidence_mean"),
                        ("+ area trends", f"{y} ~ exposure + C(area) + C(year) + C(area):t + np.log(total_chars) + rating_z + rating_sd_z + confidence_mean"),
                        ("2024-26 only", f"{y} ~ exposure + C(area) + C(year) + np.log(total_chars) + rating_z + rating_sd_z + confidence_mean")]:
            d = P if spec != "2024-26 only" else P[P.year >= 2024]
            d = d.dropna(subset=[y, "exposure", "total_chars", "rating_z", "rating_sd_z", "confidence_mean", "area"])
            fit = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.area.astype("category").cat.codes * 100 + d.year})
            rows.append(dict(outcome=y, spec=spec, coef=fit.params["exposure"], se=fit.bse["exposure"], p=fit.pvalues["exposure"], n=int(fit.nobs),
                             sd_y=d[y].std()))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / f"did_{model}{tag}.csv", index=False)
    # event study: areas split into terciles by their 2026 exposure; year x tercile coefficients relative to 2023
    dose = cell[cell.year == 2026].set_index("area").exposure
    terc = pd.qcut(dose, 3, labels=["low", "mid", "high"])
    P["dose_terc"] = P.area.map(terc)
    ev = []
    dums = []
    for yr in sorted(P.year.unique()):
        for t in ["mid", "high"]:
            if yr != 2023:
                P[f"ev_{yr}_{t}"] = ((P.year == yr) & (P.dose_terc == t)).astype(int)
                dums.append(f"ev_{yr}_{t}")
    for y in ["S3", "share_shared_claims", "D3", "pair_rate_strict"]:
        d = P.dropna(subset=["dose_terc", y])
        fit = smf.ols(f"{y} ~ C(area) + C(year) + " + " + ".join(dums), data=d).fit(cov_type="cluster", cov_kwds={"groups": d.area.astype("category").cat.codes * 100 + d.year})
        for k in dums:
            _, yr, t = k.split("_")
            ev.append(dict(outcome=y, year=int(yr), tercile=t, coef=fit.params[k], se=fit.bse[k], p=fit.pvalues[k]))
    ev = pd.DataFrame(ev)
    ev.to_csv(REPORTS / f"event_study_{model}{tag}.csv", index=False)
    print("\nEvent study (year x 2026-dose tercile vs low tercile, ref 2023); 2026 exposure terciles:", terc.to_dict())
    print(ev[ev.outcome == "S3"].pivot(index="year", columns="tercile", values="coef").round(3).to_string())
    print("\nEffect of cell exposure (0->1 = all reviews flagged); SEs clustered by area-year cell")
    for _, r in out.iterrows():
        print(f"  {r.outcome:22s} {r.spec:14s} {r.coef:+.3f} (SE {r.se:.3f}) p={r.p:.3f}  [per 10pp: {r.coef/10:+.4f}; sd_y {r.sd_y:.3f}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
