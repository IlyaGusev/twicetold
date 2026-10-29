"""E1/E2: year trends of claim-level outcomes (ICLR 2017-2026, NeurIPS 2021-2025) with the shuffled-paper placebo.

Usage: analysis_trends.py <model> [tag]
Writes reports/year_means_<model>.csv, reports/year_effects_<model>.csv, reports/placebo_<model>.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
OUTCOMES = ["D3", "S3", "U3", "T3", "TS3", "share_shared_claims", "share_shared_claims_lenient", "pair_rate_strict",
            "hit_rate_strict", "hit_rate_lenient", "claims_per_review", "hill_shannon", "share_specific"]
CONTROLS = "np.log(total_chars) + rating_z + rating_sd_z + confidence_mean + C(area)"


def year_effects(P, outcome, ref, controls):
    f = f"{outcome} ~ C(year, Treatment({ref}))" + (f" + {controls}" if controls else "")
    fit = smf.ols(f, data=P).fit(cov_type="HC3")
    rows = []
    for k, v in fit.params.items():
        if k.startswith("C(year"):
            y = int(k.split("[T.")[1].rstrip("]"))
            rows.append(dict(outcome=outcome, year=y, coef=v, se=fit.bse[k], p=fit.pvalues[k]))
    rows.append(dict(outcome=outcome, year=ref, coef=0.0, se=0.0, p=1.0))
    return pd.DataFrame(rows)


def pretrend(P, outcome, years):
    sub = P[P.year.isin(years)]
    fit = smf.ols(f"{outcome} ~ year + np.log(total_chars)", data=sub).fit(cov_type="HC3")
    return fit.params["year"], fit.bse["year"], fit.pvalues["year"]


def main(model, tag=""):
    P = pd.read_parquet(DATA / f"panel_outcomes_{model}{tag}.parquet")
    P = P[P.n_claims > 0].copy()
    P["confidence_mean"] = P.confidence_mean.fillna(P.confidence_mean.median())
    P["rating_sd"] = P.rating_sd.fillna(P.rating_sd.median())
    P["rating_mean"] = P.rating_mean.fillna(P.rating_mean.median())
    # ratings are standardized within venue-year because the ICLR 2026 scale changed
    g = P.groupby(["conf", "year"])
    P["rating_z"] = (P.rating_mean - g.rating_mean.transform("mean")) / g.rating_mean.transform("std")
    P["rating_sd_z"] = P.rating_sd / g.rating_sd.transform("mean")
    P["area"] = P.area.fillna("other")
    REPORTS.mkdir(exist_ok=True)

    means = P.groupby(["conf", "year"])[OUTCOMES + ["n_reviews", "total_chars"]].mean()
    means["papers"] = P.groupby(["conf", "year"]).size()
    means.round(3).to_csv(REPORTS / f"year_means_{model}{tag}.csv")
    print(f"== {model}{tag}: {len(P)} panels\nRaw means by venue-year")
    print(means[["papers", "claims_per_review", "D3", "S3", "U3", "T3", "share_shared_claims", "pair_rate_strict", "hit_rate_strict", "share_specific"]].round(3).to_string())

    effects = []
    for conf, ref in [("ICLR", 2023), ("NeurIPS", 2022)]:
        sub = P[P.conf == conf]
        if ref not in set(sub.year):
            continue
        for spec, controls in [("raw", ""), ("length", "np.log(total_chars)"), ("full", CONTROLS)]:
            for y in OUTCOMES:
                e = year_effects(sub, y, ref, controls)
                e["conf"], e["spec"] = conf, spec
                effects.append(e)
    # robustness specs, ICLR only: panels with no edited review; area mix reweighted to 2017-2022; specific-only and
    # claim-count-matched outcomes (raw)
    sub = P[P.conf == "ICLR"]
    pre_mix = sub[sub.year.between(2017, 2022)].area.value_counts(normalize=True)
    w = pre_mix.reindex(sub.area).to_numpy() / sub.groupby("year").area.transform(lambda a: a.value_counts(normalize=True).reindex(a).to_numpy())
    for spec, d, controls, ws in [("unedited", sub[(sub.n_edited == 0) & (sub.n_rebuttal_text == 0)], "", None),
                                  ("area-reweighted", sub, "", w)]:
        for y in OUTCOMES:
            if ws is None:
                e = year_effects(d, y, 2023, controls)
            else:
                fit = smf.wls(f"{y} ~ C(year, Treatment(2023))", data=d, weights=ws).fit(cov_type="HC3")
                e = pd.DataFrame([dict(outcome=y, year=int(k.split("[T.")[1].rstrip("]")), coef=v, se=fit.bse[k], p=fit.pvalues[k])
                                  for k, v in fit.params.items() if k.startswith("C(year")])
            e["conf"], e["spec"] = "ICLR", spec
            effects.append(e)
    for y in ["S3_spec", "share_shared_claims_spec", "D3_spec", "S3_m8", "share_shared_claims_m8", "D3_m8"]:
        if y in sub:
            e = year_effects(sub, y, 2023, "")
            e["conf"], e["spec"] = "ICLR", "raw"
            effects.append(e)
    effects = pd.concat(effects)
    effects.to_csv(REPORTS / f"year_effects_{model}{tag}.csv", index=False)

    print("\nICLR year effects vs 2023, full controls (coef (SE)):")
    tab = effects[(effects.conf == "ICLR") & (effects.spec == "full")]
    tab = tab.assign(txt=tab.apply(lambda r: f"{r.coef:+.3f} ({r.se:.3f})", axis=1)).pivot(index="outcome", columns="year", values="txt")
    print(tab.to_string())

    print("\nPre-trend 2017-2023 (ICLR), slope per year with length control:")
    for y in ["D3", "S3", "U3", "share_shared_claims", "pair_rate_strict", "claims_per_review"]:
        b, se, p = pretrend(P[P.conf == "ICLR"], y, range(2017, 2024))
        print(f"  {y:22s} {b:+.4f} (SE {se:.4f}, p={p:.3f})")

    pl = DATA / f"placebo_outcomes_{model}{tag}.parquet"
    if pl.exists():
        Q = pd.read_parquet(pl)
        Q = Q[Q.n_claims > 0]
        if "placebo_kind" not in Q:
            Q["placebo_kind"] = "shuffled4"
        cols = ["S3", "U3", "D3", "T3", "share_shared_claims", "share_shared_claims_lenient", "pair_rate_strict", "hit_rate_strict"]
        pm = Q.groupby(["placebo_kind", "conf", "year"])[cols].mean()
        pm["panels"] = Q.groupby(["placebo_kind", "conf", "year"]).size()
        pm.round(4).to_csv(REPORTS / f"placebo_{model}{tag}.csv")
        print("\nPlacebo panels (shuffled4: 4 reviews of 4 papers; same_area3: 3 reviews of 3 same-area papers):")
        print(pm.round(4).to_string())
        # placebo-adjusted series: observed minus same-year placebo, SE from both samples
        rows = []
        for (kind, conf, year), g in Q.groupby(["placebo_kind", "conf", "year"]):
            obs = P[(P.conf == conf) & (P.year == year)]
            for y in ["S3", "share_shared_claims", "pair_rate_strict", "hit_rate_strict"]:
                rows.append(dict(placebo_kind=kind, conf=conf, year=year, outcome=y, observed=obs[y].mean(), placebo=g[y].mean(),
                                 excess=obs[y].mean() - g[y].mean(),
                                 se=np.sqrt(obs[y].var() / len(obs) + g[y].var() / len(g))))
        adj = pd.DataFrame(rows)
        adj.to_csv(REPORTS / f"placebo_adjusted_{model}{tag}.csv", index=False)
        print("\nObserved minus placebo (excess overlap), by year:")
        t = adj.assign(txt=adj.apply(lambda r: f"{r.excess:+.3f} ({r.se:.3f})", axis=1))
        print(t.pivot_table(index=["placebo_kind", "conf", "year"], columns="outcome", values="txt", aggfunc="first").to_string())

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
