"""Compare the pipeline's merge decisions with the independent claim-pair adjudications.

Usage: validate_pairs.py
Reads data/pairs/pairs_key.parquet (pipeline stratum, year, detector flags) and data/pairs/judgments_{a,b}{0..3}.jsonl.
Reports agreement between the two adjudication runs, the pipeline's precision/recall against them by period and by
flag stratum, and the Rogan-Gladen corrected merge rate.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

P = Path(__file__).resolve().parent.parent / "data" / "pairs"
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def load(run):
    rows = []
    for k in range(4):
        f = P / f"judgments_{run}{k}.jsonl"
        if not f.exists():
            continue
        # keep only judgments whose pair_id belongs to this input file (agents shared a scratchpad; guard against mix-ups)
        ids = {json.loads(l)["pair_id"] for l in open(P / f"pairs_input_{k}.jsonl")}
        js = [json.loads(l) for l in open(f) if l.strip()]
        ok = [j for j in js if j["pair_id"] in ids]
        print(f"judgments_{run}{k}: {len(ok)} valid of {len(js)}")
        rows += ok
    return pd.DataFrame(rows).drop_duplicates("pair_id", keep="last").set_index("pair_id")


def main():
    K = pd.read_parquet(P / "pairs_key.parquet").set_index("pair_id")
    K["period"] = np.where(K.year <= 2022, "pre", "post")
    K["flagged"] = K.a_flag | K.b_flag
    K["pipe_problem"] = K.stratum == "same_cluster"
    K["pipe_target"] = K.stratum != "different"
    A, B = load("a"), load("b")
    both = K.join(A.add_suffix("_a"), how="inner").join(B.add_suffix("_b"), how="inner")
    print(f"{len(both)} pairs with both adjudications")
    for q in ["same_problem", "same_target"]:
        print(f"  run agreement on {q}: {(both[q+'_a'] == both[q+'_b']).mean():.3f}, kappa {cohen_kappa_score(both[q+'_a'], both[q+'_b']):.3f}")
    both["ref_problem"] = both.same_problem_a & both.same_problem_b
    both["ref_problem_any"] = both.same_problem_a | both.same_problem_b
    both["ref_target"] = both.same_target_a & both.same_target_b
    rows = []
    for name, g in [("all", both), ("pre", both[both.period == "pre"]), ("post", both[both.period == "post"]),
                    ("post, unflagged", both[(both.period == "post") & ~both.flagged]), ("post, flagged", both[(both.period == "post") & both.flagged])]:
        for ref in ["ref_problem", "ref_problem_any"]:
            tp = (g.pipe_problem & g[ref]).sum(); fp = (g.pipe_problem & ~g[ref]).sum(); fn = (~g.pipe_problem & g[ref]).sum(); tn = (~g.pipe_problem & ~g[ref]).sum()
            rows.append(dict(subset=name, reference=ref, n=len(g), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                             fpr=fp / max(fp + tn, 1), n_pipe_merge=int(g.pipe_problem.sum()), n_ref_merge=int(g[ref].sum())))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "pairs_validation.csv", index=False)
    print(out.round(3).to_string())
    # within each stratum, the share of pipeline-merged pairs that the adjudicators also merge (precision), by period/flag
    print("\nPrecision of pipeline merges (both adjudicators agree) by stratum:")
    print(both[both.pipe_problem].groupby(["period", "flagged"]).ref_problem.agg(["mean", "size"]).round(3).to_string())
    print("\nRecall: share of adjudicated same-problem pairs that the pipeline merged, by stratum (sampling weights ignored):")
    print(both[both.ref_problem].groupby(["period", "flagged"]).pipe_problem.agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
