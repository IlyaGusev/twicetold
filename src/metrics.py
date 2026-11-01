import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from extract import CATEGORIES

MATCHED_CLAIMS_PER_REVIEW = 8
MATCHED_SUFFIX = f"m{MATCHED_CLAIMS_PER_REVIEW}"


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


def rarefy_groups(groups, ks):
    out = {}
    for k in ks:
        if k > len(groups):
            out[k] = (np.nan, np.nan, np.nan)
            continue
        sizes, shared, unique = [], [], []
        for subset in combinations(groups, k):
            counts = Counter(group for review in subset for group in review)
            sizes.append(len(counts))
            shared.append(sum(n >= 2 for n in counts.values()))
            if counts:
                unique.append(sum(n == 1 for n in counts.values()) / len(counts))
        out[k] = (sum(sizes) / len(sizes), sum(shared) / len(shared), sum(unique) / len(unique) if unique else np.nan)
    return out


def overlap(groups_a, groups_b):
    groups_a, groups_b = set(groups_a), set(groups_b)
    denominator = min(len(groups_a), len(groups_b))
    return len(groups_a & groups_b) / denominator if denominator else np.nan


def mean_overlap(groups):
    rates = [overlap(a, b) for a, b in combinations([group for group in groups if group], 2)]
    return sum(rates) / len(rates) if rates else np.nan


def shared_fraction(groups, values):
    counts = Counter(group for review in groups for group in review)
    return sum(counts[group] >= 2 for group in values) / len(values) if len(values) else np.nan


def subset_metrics(reviews, values, suffix, n_reviews):
    groups = review_groups(reviews, values, n_reviews)
    d3, s3, _ = rarefy_groups(groups, (3,))[3]
    return {f"S3_{suffix}": s3, f"D3_{suffix}": d3, f"share_shared_claims_{suffix}": shared_fraction(groups, values)}


def panel_metrics(df, n_reviews, ks=(1, 2, 3, 4)):
    reviews = df.review.to_numpy(dtype=int) if not df.empty else np.array([], dtype=int)
    clusters = df.cluster.to_numpy() if not df.empty else np.array([], dtype=int)
    targets = df.target.to_numpy() if not df.empty else np.array([], dtype=int)
    groups = review_groups(reviews, clusters, n_reviews)
    target_groups = review_groups(reviews, targets, n_reviews)
    cluster_reviewers = Counter(group for review in groups for group in review)
    target_reviewers = Counter(group for review in target_groups for group in review)
    metrics = dict(
        n_reviews=n_reviews,
        n_claims=len(df),
        n_clusters=len(cluster_reviewers),
        n_targets=len(target_reviewers),
        n_shared_clusters=sum(n >= 2 for n in cluster_reviewers.values()),
        n_shared_targets=sum(n >= 2 for n in target_reviewers.values()),
        claims_per_review=len(df) / n_reviews,
    )
    for prefix, source in [("", groups), ("T", target_groups)]:
        for k, (n_groups, n_shared, unique) in rarefy_groups(source, ks).items():
            if prefix:
                metrics.update({f"T{k}": n_groups, f"TS{k}": n_shared, f"TU{k}": unique})
            else:
                metrics.update({f"D{k}": n_groups, f"S{k}": n_shared, f"U{k}": unique})
    if df.empty:
        metrics.update(subset_metrics(reviews, clusters, "spec", n_reviews))
        metrics.update(subset_metrics(reviews, clusters, MATCHED_SUFFIX, n_reviews))
        return metrics
    weights = np.array(list(cluster_reviewers.values()), dtype=float)
    weights /= weights.sum()
    specific = df.specific.to_numpy(dtype=bool)
    category_counts = Counter(df.category)
    metrics.update(
        share_shared_claims=shared_fraction(groups, clusters),
        share_shared_claims_lenient=shared_fraction(target_groups, targets),
        share_unique_clusters=sum(n == 1 for n in cluster_reviewers.values()) / len(cluster_reviewers),
        hill_shannon=float(np.exp(-(weights * np.log(weights)).sum())),
        share_specific=float(specific.mean()),
        hit_rate_strict=mean_overlap(groups),
        hit_rate_lenient=mean_overlap(target_groups),
        **{f"cat_{category}": category_counts[category] / len(df) for category in CATEGORIES},
    )
    metrics.update(subset_metrics(reviews[specific], clusters[specific], "spec", n_reviews))
    taken = Counter()
    indices = []
    for i in np.random.RandomState(0).choice(len(df), len(df), replace=False):
        if taken[reviews[i]] < MATCHED_CLAIMS_PER_REVIEW:
            indices.append(i)
            taken[reviews[i]] += 1
    metrics.update(subset_metrics(reviews[indices], clusters[indices], MATCHED_SUFFIX, n_reviews))
    return metrics


def score_panel(panel, claims_by_review, ks=(1, 2, 3, 4)):
    return panel_metrics(assign(panel, claims_by_review), len(panel["review_ids"]), ks)
