"""Run the two-pass pipeline on the 30 blind pilot papers (weakness text only) and compare with the Opus reference.

Usage: validate_pilot.py prepare | compare
Between the two steps run:  MODEL=<m> TAG=_pilot uv run python -I src/extract.py pass1 data/pilot_reviews.parquet
                            MODEL=<m> TAG=_pilot uv run python -I src/extract.py pass2 data/pilot_panels.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from scipy.stats import spearmanr

from metrics import all_panel_metrics

PRELIM = Path("/home/yallen/ideas/reports/peer-review-prelim/pilot")
DATA = Path(__file__).resolve().parent.parent / "data"
METRICS = {"n_claims": "n_claims", "n_clusters": "n_clusters", "n_shared_clusters": "n_shared_clusters",
           "share_generic": "share_generic", "hill_shannon": "hill_shannon"}


def prepare():
    rows, panels = [], []
    for line in open(PRELIM / "pilot_input.jsonl"):
        it = json.loads(line)
        ids = []
        for r in it["reviews"]:
            rid = f"{it['paper']}_{r['review']}"
            rows.append(dict(review_id=rid, text=f"[WEAKNESSES] {r['weaknesses']}"))
            ids.append(rid)
        panels.append(dict(panel_id=it["paper"], review_ids=ids))
    pd.DataFrame(rows).to_parquet(DATA / "pilot_reviews.parquet")
    with open(DATA / "pilot_panels.jsonl", "w") as f:
        for p in panels:
            f.write(json.dumps(p) + "\n")
    print(len(rows), "reviews,", len(panels), "panels")


def compare(models, tag="_pilot"):
    opus = pd.read_csv(PRELIM / "pilot_metrics.csv").set_index("paper")
    key = {k: v["year"] for k, v in json.load(open(PRELIM / "pilot_key.json")).items()}
    for model in models:
        m = all_panel_metrics(DATA / f"claims_{model}{tag}.jsonl", DATA / f"clusters_{model}{tag}.jsonl").set_index("panel_id")
        m["share_generic"] = 1 - m.share_specific
        m["year"] = m.index.map(key)
        print(f"\n== {model} vs Opus reference ({len(m)} papers)")
        for k_opus, k_new in METRICS.items():
            both = pd.concat([opus[k_opus], m[k_new]], axis=1, keys=["opus", "new"]).dropna()
            rho = spearmanr(both.opus, both.new).statistic
            print(f"  {k_new:20s} opus {both.opus.mean():6.2f}  new {both.new.mean():6.2f}  spearman {rho:.2f}")
        print("  by year:", m.groupby("year")[["n_claims", "n_clusters", "n_shared_clusters", "share_shared_claims", "D3", "T3"]].mean().round(2).to_dict("index"))
        old = PRELIM / f"llm_validate_{model}_metrics.csv"
        if old.exists():
            o = pd.read_csv(old).set_index("paper")
            print("  one-pass pilot (same model) vs Opus:",
                  {k: round(spearmanr(opus[k], o[k]).statistic, 2) for k in ["n_claims", "n_clusters", "n_shared_clusters"]})


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        compare(sys.argv[3:], sys.argv[2])
