"""Claim-pair adjudication set for calibrating the pass-2 merge decision.

Samples cross-review claim pairs from ICLR 2017-2022 and ICLR 2026 panels, stratified by the pipeline's decision
(same cluster / same target but different cluster / different target). Writes data/pairs/pairs_input_{k}.jsonl
(pair id, the two claims and the two source reviews' field-tagged text windows) and data/pairs/pairs_key.parquet
(pipeline decision and year, hidden from annotators).

Usage: make_pairs.py <model>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from metrics import assign, load_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "pairs"
PER_STRATUM = 100  # x 3 strata x 2 periods = 600 pairs
rng = np.random.default_rng(3)


def main(model):
    claims = {k: v["claims"] for k, v in load_jsonl(DATA / f"claims_{model}.jsonl", "review_id").items()}
    panels = load_jsonl(DATA / f"clusters_{model}.jsonl", "panel_id")
    meta = pd.read_parquet(DATA / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    texts = pd.read_parquet(DATA / "sample_reviews.parquet", columns=["review_id", "text"]).set_index("review_id").text
    flag = pd.read_parquet(DATA / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag
    # post-period pairs are split by whether either review is flagged as LLM-styled (half each)
    pools = {(per, s): [] for per in ("pre", "post_flag", "post_noflag") for s in ("same_cluster", "same_target", "different")}
    for pid, p in panels.items():
        if pid not in meta.index or meta.loc[pid, "conf"] != "ICLR":
            continue
        year = int(meta.loc[pid, "year"])
        period = "pre" if year <= 2022 else "post" if year == 2026 else None
        if period is None:
            continue
        df = assign(p, claims)
        if df.empty:
            continue
        a = df.sample(frac=1, random_state=int(rng.integers(1e9)))
        seen = 0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                x, y = a.iloc[i], a.iloc[j]
                if x.review == y.review:
                    continue
                s = "same_cluster" if x.cluster == y.cluster else "same_target" if x.target == y.target else "different"
                if s == "different" and rng.random() > 0.02:
                    continue
                fa, fb = bool(flag.get(x.review_id, False)), bool(flag.get(y.review_id, False))
                key = period if period == "pre" else ("post_flag" if fa or fb else "post_noflag")
                pools[(key, s)].append(dict(paper_id=pid, year=year, stratum=s, a_review=x.review_id, b_review=y.review_id,
                                            a_flag=fa, b_flag=fb, a_claim=x.claim, b_claim=y.claim, a_cat=x.category, b_cat=y.category))
                seen += 1
                if seen > 30:
                    break
            if seen > 30:
                break
    rows = []
    for key, pool in pools.items():
        print(key, len(pool))
        n = PER_STRATUM if key[0] == "pre" else PER_STRATUM // 2
        idx = rng.choice(len(pool), min(n, len(pool)), replace=False)
        rows += [pool[i] for i in idx]
    rng.shuffle(rows)
    K = pd.DataFrame(rows)
    K["pair_id"] = [f"Q{i:03d}" for i in range(len(K))]
    OUT.mkdir(exist_ok=True)
    K.to_parquet(OUT / "pairs_key.parquet")
    items = [dict(pair_id=r.pair_id, claim_a=r.a_claim, claim_b=r.b_claim,
                  review_a=texts.get(r.a_review, "")[:6000], review_b=texts.get(r.b_review, "")[:6000]) for r in K.itertuples()]
    for k in range(4):
        with open(OUT / f"pairs_input_{k}.jsonl", "w") as f:
            for it in items[k::4]:
                f.write(json.dumps(it) + "\n")
    print(len(K), "pairs;", K.groupby(["year", "stratum"]).size().to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
