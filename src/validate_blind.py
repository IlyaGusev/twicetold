"""Compare the pipeline with the blind Opus reference on 40 papers (10 each from ICLR 2018, 2022, 2024, 2026).

Usage: validate_blind.py <model> [tag]
Reports Spearman agreement per outcome, mean bias by year (the year-dependence check), and claim-level recall.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from metrics import assign, load_jsonl, panel_metrics

DATA = Path(__file__).resolve().parent.parent / "data"
BLIND = DATA / "blind"
OUTS = ["n_claims", "n_clusters", "n_targets", "n_shared_clusters", "share_shared_claims", "share_shared_claims_lenient",
        "D3", "S3", "U3", "T3", "pair_rate_strict", "hit_rate_strict", "share_specific", "hill_shannon"]


def opus_metrics(ks=range(4)):
    claims, panels = {}, {}
    for k in ks:
        if not (BLIND / f"opus_claims_{k}.jsonl").exists():
            continue
        claims.update({r["review_id"]: r["claims"] for r in map(json.loads, open(BLIND / f"opus_claims_{k}.jsonl"))})
        panels.update({p["panel_id"]: p for p in map(json.loads, open(BLIND / f"opus_clusters_{k}.jsonl"))})
    rows = []
    for pid, p in panels.items():
        rows.append(dict(panel_id=pid, **panel_metrics(assign(p, claims))))
    return pd.DataFrame(rows).set_index("panel_id")


def model_metrics(model, tag):
    key = json.load(open(BLIND / "blind_key.json"))
    claims = {k: v["claims"] for k, v in load_jsonl(DATA / f"claims_{model}{tag}.jsonl", "review_id").items()}
    panels = load_jsonl(DATA / f"clusters_{model}{tag}.jsonl", "panel_id")
    rows = []
    for code, v in key.items():
        p = panels.get(code)
        if p is None:
            continue
        rows.append(dict(panel_id=code, year=v["year"], **panel_metrics(assign(p, claims))))
    return pd.DataFrame(rows).set_index("panel_id")


def reference_agreement():
    """Agreement between the original Opus reference and an independent Opus replicate on the same papers:
    the noise ceiling for the pipeline-vs-reference comparison."""
    o, r = opus_metrics(), opus_metrics(["rep0", "rep1"])
    both = o.join(r, how="inner", lsuffix="_a", rsuffix="_b")
    if both.empty:
        return
    print(f"== Opus reference vs independent Opus replicate: {len(both)} papers")
    for k in OUTS:
        a, b = both[f"{k}_a"], both[f"{k}_b"]
        d = b - a
        print(f"{k:30s} {a.mean():7.3f} {b.mean():7.3f} rho {spearmanr(a, b).statistic:5.2f}  bias {d.mean():+.3f} (SE {d.std(ddof=1)/np.sqrt(len(d)):.3f})")


def main(model, tag=""):
    reference_agreement()
    o, m = opus_metrics(), model_metrics(model, tag)
    both = o.join(m, how="inner", lsuffix="_opus", rsuffix="_model")
    print(f"== {model}{tag} vs blind Opus reference: {len(both)} papers")
    print(f"{'outcome':30s} {'opus':>7s} {'model':>7s} {'rho':>6s}")
    for k in OUTS:
        a, b = both[f"{k}_opus"], both[f"{k}_model"]
        print(f"{k:30s} {a.mean():7.3f} {b.mean():7.3f} {spearmanr(a, b).statistic:6.2f}")
    print("\nBias (model - opus) by year, with SE:")
    for k in ["n_claims", "n_clusters", "n_shared_clusters", "share_shared_claims", "S3", "U3", "hit_rate_strict"]:
        d = both[f"{k}_model"] - both[f"{k}_opus"]
        g = d.groupby(both.year)
        print(f"  {k:22s}", {y: f"{v.mean():+.3f} ({v.std(ddof=1)/np.sqrt(len(v)):.3f})" for y, v in g})
    print("\nEquivalence (TOST) of year bias: p that |bias(2026) - bias(2018-2022)| < bound")
    for k, bound in [("share_shared_claims", 0.03), ("S3", 0.3), ("hit_rate_strict", 0.03)]:
        d = both[f"{k}_model"] - both[f"{k}_opus"]
        late, early = d[both.year == 2026], d[both.year.isin([2018, 2022])]
        diff = late.mean() - early.mean()
        se = np.sqrt(late.var(ddof=1) / len(late) + early.var(ddof=1) / len(early))
        p = max(1 - norm.cdf((diff + bound) / se), 1 - norm.cdf((bound - diff) / se))
        print(f"  {k:22s} diff {diff:+.3f} (SE {se:.3f}), bound {bound}, TOST p={p:.3f}")
    print("\nMeans by year (opus | model) for S3 and share_shared_claims:")
    for k in ["S3", "share_shared_claims", "D3", "n_claims"]:
        print(f"  {k:22s}", both.groupby("year")[[f"{k}_opus", f"{k}_model"]].mean().round(3).to_dict("index"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
