"""ICML 2026 policy comparison (RQ4, observational): panel outcomes of papers whose authors required Policy A (no LLM)
against papers that allowed Policy B (understand and polish), with primary-area fixed effects; detector flagged share
by policy as the manipulation check. Needs data/icml2026_reviews.parquet (fetch_icml2026.py), the pipeline run with
MODEL=gpt-6-luna TAG=_icml on it (panels data/panels_icml.jsonl, prefix icml_), outcomes.py gpt-6-luna _icml and
detector scores for the ICML reviews. Usage: analysis_icml.py <model> -> reports/icml_policy_<model>.csv"""

import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
OUTS = ["S3", "D3", "U3", "share_shared_claims", "hit_rate_strict", "claims_per_review", "share_specific"]


def main(model):
    R = pd.read_parquet(ROOT / "data" / "icml2026_reviews.parquet")
    P = pd.read_parquet(ROOT / "data" / f"panel_outcomes_{model}_icml.parquet")
    P["paper_id"] = P.panel_id.str.replace("^icml_", "", regex=True)
    meta = R.groupby("paper_id").agg(llm_policy=("llm_policy", "first"), primary_area=("primary_area", "first"))
    P = P.join(meta, on="paper_id")
    P["permissive"] = P.llm_policy.str.contains("B", na=False).astype(int)
    rows = []
    for out in OUTS:
        m = smf.ols(f"{out} ~ permissive + C(primary_area)", data=P).fit(cov_type="HC1")
        rows.append(dict(outcome=out, mean_A=P[P.permissive == 0][out].mean(), mean_B=P[P.permissive == 1][out].mean(),
                         coef=m.params.permissive, se=m.bse.permissive, n=int(m.nobs)))
    det = ROOT / "data" / "detector_scores_icml.parquet"
    if det.exists():
        D = pd.read_parquet(det).merge(R[["review_id", "llm_policy"]], on="review_id")
        print("flagged share by policy:", D.groupby("llm_policy").llm_flag.mean().round(3).to_dict())
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / "reports" / f"icml_policy_{model}.csv", index=False)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
