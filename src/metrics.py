import json
from itertools import combinations

import numpy as np
import pandas as pd


def load_jsonl(path, key):
    with open(path) as f:
        rows = map(json.loads, f)
        return {row[key]: row for row in rows if "error" not in row}


def assign(panel, claims_by_review):
    rows = []
    for review, rid in enumerate(panel["review_ids"]):
        for k, claim in enumerate(claims_by_review.get(rid, [])):
            rows.append(dict(cid=f"R{review}-{k}", review=review, review_id=rid, **claim))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    clusters = panel.get("clusters", [])
    cluster_of, target_of = {}, {}
    for cluster_id, cluster in enumerate(clusters):
        for cid in cluster["claim_ids"]:
            cluster_of.setdefault(cid, cluster_id)
            target_of.setdefault(cid, cluster["target_id"])
    next_cluster = len(clusters)
    next_target = max(target_of.values(), default=0) + 1
    for cid in df.cid:
        if cid not in cluster_of:
            cluster_of[cid] = next_cluster
            target_of[cid] = next_target
            next_cluster += 1
            next_target += 1
    df["cluster"] = df.cid.map(cluster_of)
    df["target"] = df.cid.map(target_of)
    return df


def review_groups(reviews, values, n_reviews):
    groups = [set() for _ in range(n_reviews)]
    for review, value in zip(reviews, values):
        groups[review].add(value)
    return groups


def overlap(groups_a, groups_b):
    groups_a, groups_b = set(groups_a), set(groups_b)
    denominator = min(len(groups_a), len(groups_b))
    return len(groups_a & groups_b) / denominator if denominator else 0.0


def mean_overlap(groups):
    rates = [overlap(a, b) for a, b in combinations(groups, 2)]
    return sum(rates) / len(rates) if rates else np.nan


def panel_metrics(df, n_reviews):
    def groups(column):
        return review_groups([], [], n_reviews) if df.empty else review_groups(df.review, df[column], n_reviews)

    return dict(
        n_reviews=n_reviews,
        n_claims=len(df),
        claims_per_review=len(df) / n_reviews,
        hit_rate_strict=mean_overlap(groups("cluster")),
        hit_rate_lenient=mean_overlap(groups("target")),
    )
