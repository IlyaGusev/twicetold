import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from extract import CATEGORIES

MATCHED_CLAIMS_PER_REVIEW = 8


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


def rarefy(df, col, ks):
    groups_of = df.groupby("review")[col].agg(set).to_dict()
    reviews = sorted(groups_of)
    out = {}
    for k in ks:
        if k > len(reviews):
            out[k] = (np.nan, np.nan, np.nan)
            continue
        n_groups, n_shared, unique_share = [], [], []
        for subset in combinations(reviews, k):
            reviews_per_group = Counter(group for review in subset for group in groups_of[review])
            n_groups.append(len(reviews_per_group))
            n_shared.append(sum(n >= 2 for n in reviews_per_group.values()))
            unique_share.append(np.mean([n == 1 for n in reviews_per_group.values()]) if reviews_per_group else np.nan)
        out[k] = (float(np.mean(n_groups)), float(np.mean(n_shared)), float(np.nanmean(unique_share)))
    return out


def pair_rates(df, col):
    reviews = sorted(df.review.unique())
    n_same = n_compared = 0
    hit_rates = []
    for a, b in combinations(reviews, 2):
        groups_a = df[df.review == a][col].to_numpy()
        groups_b = df[df.review == b][col].to_numpy()
        same_group = groups_a[:, None] == groups_b[None, :]
        n_same += int(same_group.sum())
        n_compared += same_group.size
        hit_rates.append(same_group.any(axis=1).mean())
        hit_rates.append(same_group.any(axis=0).mean())
    pair_rate = n_same / n_compared if n_compared else np.nan
    hit_rate = float(np.mean(hit_rates)) if hit_rates else np.nan
    return pair_rate, hit_rate


def share_shared_claims(df, col):
    reviewers = df.groupby(col).review.nunique()
    return float(df[col].map(reviewers).ge(2).mean())


def _subset_metrics(df, suffix):
    if df.empty:
        d3 = s3 = shared = np.nan
    else:
        shared = share_shared_claims(df, "cluster")
        d3, s3, _ = rarefy(df, "cluster", (3,))[3]
    return {f"S3_{suffix}": s3, f"D3_{suffix}": d3, f"share_shared_claims_{suffix}": shared}


def panel_metrics(df, ks=(1, 2, 3, 4)):
    if df.empty:
        return dict(n_reviews=0, n_claims=0)
    n_reviews = df.review.nunique()
    cluster_reviewers = df.groupby("cluster").review.nunique()
    target_reviewers = df.groupby("target").review.nunique()
    cluster_weights = cluster_reviewers / cluster_reviewers.sum()
    metrics = dict(
        n_reviews=n_reviews,
        n_claims=len(df),
        n_clusters=int(cluster_reviewers.size),
        n_targets=int(target_reviewers.size),
        n_shared_clusters=int((cluster_reviewers >= 2).sum()),
        n_shared_targets=int((target_reviewers >= 2).sum()),
        share_shared_claims=share_shared_claims(df, "cluster"),
        share_shared_claims_lenient=share_shared_claims(df, "target"),
        share_unique_clusters=float((cluster_reviewers == 1).mean()),
        hill_shannon=float(np.exp(-(cluster_weights * np.log(cluster_weights)).sum())),
        share_specific=float(df.specific.mean()),
        claims_per_review=len(df) / n_reviews,
    )
    metrics["pair_rate_strict"], metrics["hit_rate_strict"] = pair_rates(df, "cluster")
    metrics["pair_rate_lenient"], metrics["hit_rate_lenient"] = pair_rates(df, "target")
    for k, (n_groups, n_shared, unique_share) in rarefy(df, "cluster", ks).items():
        metrics.update({f"D{k}": n_groups, f"S{k}": n_shared, f"U{k}": unique_share})
    for k, (n_groups, n_shared, unique_share) in rarefy(df, "target", ks).items():
        metrics.update({f"T{k}": n_groups, f"TS{k}": n_shared, f"TU{k}": unique_share})
    for category in CATEGORIES:
        metrics[f"cat_{category}"] = float((df.category == category).mean())
    specific_claims = df[df.specific.astype(bool)]
    metrics.update(_subset_metrics(specific_claims, "spec"))
    matched = df.sample(frac=1, random_state=0).groupby("review").head(MATCHED_CLAIMS_PER_REVIEW)
    metrics.update(_subset_metrics(matched, f"m{MATCHED_CLAIMS_PER_REVIEW}"))
    return metrics
