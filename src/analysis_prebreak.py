"""Placebo break at 2021 (E1): a level shift for 2021-2022 against a linear trend fitted inside the pre-period,
for three pre-period windows, next to the same model for the real 2026 break. -> reports/prebreak_<model>.csv"""

import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
OUTS = ["S3", "share_shared_claims", "hit_rate_strict", "D3"]


def main(model):
    P = pd.read_parquet(ROOT / "data" / f"panel_outcomes_{model}.parquet")
    P = P[(P.n_claims > 0) & (P.conf == "ICLR") & ~P.panel_id.str.contains("placebo|sub3|noq|wd_")]
    rows = []
    for lo, hi, brk in [(2017, 2022, 2021), (2018, 2022, 2021), (2019, 2022, 2021), (2020, 2026, 2026)]:
        d = P[P.year.between(lo, hi) & (P.year != 2024)].copy()  # 2024 form change dropped from the 2026 window
        d["post"], d["t"] = (d.year >= brk).astype(int), d.year - lo
        for out in OUTS:
            m = smf.ols(f"{out} ~ t + post", data=d).fit(cov_type="HC1")
            rows.append(dict(window=f"{lo}-{hi}", break_year=brk, outcome=out, coef=m.params.post, se=m.bse.post, n=len(d)))
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / "reports" / f"prebreak_{model}.csv", index=False)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
