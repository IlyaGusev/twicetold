"""E5: all-LLM and mixed panels versus human panels of the same papers (IntelLabs reviews, ICLR 2022-2024).

Usage: analysis_llm.py <model>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from metrics import assign, load_jsonl, panel_metrics

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
COLS = ["claims_per_review", "D3", "S3", "U3", "D4", "S4", "U4", "share_shared_claims", "pair_rate_strict", "hit_rate_strict",
        "hit_rate_lenient", "share_specific", "hill_shannon"]


def main(model):
    claims = {}
    for name in [f"claims_{model}.jsonl", f"claims_{model}_llm.jsonl"]:
        claims.update({k: v["claims"] for k, v in load_jsonl(DATA / name, "review_id").items()})
    panels = load_jsonl(DATA / f"clusters_{model}_llm.jsonl", "panel_id")
    meta = pd.read_parquet(DATA / "papers.parquet", columns=["paper_id", "year"]).set_index("paper_id").year
    rows = []
    for pid, p in panels.items():
        kind, paper = pid.split("_", 1)
        rows.append(dict(panel_id=pid, kind=kind, paper_id=paper, year=int(meta.get(paper, 0)), **panel_metrics(assign(p, claims))))
    M = pd.DataFrame(rows)
    M = M[M.n_claims > 0]
    M.to_parquet(DATA / f"llm_panel_outcomes_{model}.parquet")
    # keep only papers with all three panel kinds, for a paired comparison
    full = M.groupby("paper_id").kind.nunique()
    M3 = M[M.paper_id.isin(full[full == 3].index)]
    print(f"{len(M)} panels; {M3.paper_id.nunique()} papers with all three panel kinds")
    tab = M3.groupby(["year", "kind"])[COLS].mean().round(3)
    print(tab.to_string())
    print("\nPooled:")
    print(M3.groupby("kind")[COLS].mean().round(3).to_string())
    tab.to_csv(REPORTS / f"llm_panels_{model}.csv")
    M3.groupby("kind")[COLS].mean().to_csv(REPORTS / f"llm_panels_pooled_{model}.csv")
    hum = M3[M3.kind == "hum4"].set_index("paper_id")
    llm = M3[M3.kind == "llm4"].set_index("paper_id")
    d = (llm[COLS] - hum[COLS].reindex(llm.index))
    print("\nPaired difference llm4 - hum4 (mean, SE):")
    print(pd.DataFrame(dict(mean=d.mean(), se=d.std() / len(d) ** 0.5)).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
