"""Panel-level outcomes from pass-1 claims and pass-2 clusters (library, no CLI; import with src/ on sys.path).

assign() turns one pass-2 panel into a claim table with strict cluster and lenient target ids; panel_metrics() computes
the panel outcomes (shared claims, rarefied distinct/shared counts, pair and hit rates, category shares, variants).
"""

import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from extract import CATEGORIES

M_MATCH = 8  # claim-count matched variant: at most this many claims per review


def load_jsonl(path, key):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            out[r[key]] = r
    return out


def assign(panel, claims_by_review):
    """Return a DataFrame of claims with review index, strict cluster id and lenient target id.

    Claims the model left out become singleton clusters; a claim listed twice keeps its first assignment.
    """
    rows = []
    for j, rid in enumerate(panel["review_ids"]):
        for k, c in enumerate(claims_by_review.get(rid, [])):
            rows.append(dict(cid=f"R{j}-{k}", review=j, review_id=rid, **c))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cluster, target = {}, {}
    for n, cl in enumerate(panel.get("clusters", [])):
        for cid in cl["claim_ids"]:
            cluster.setdefault(cid, n)
            target.setdefault(cid, cl["target_id"])
    next_id = len(panel.get("clusters", []))
    next_target = max(target.values(), default=0) + 1
    for cid in df.cid:
        if cid not in cluster:
            cluster[cid], target[cid] = next_id, next_target
            next_id += 1
            next_target += 1
    df["cluster"] = df.cid.map(cluster)
    df["target"] = df.cid.map(target)
    return df


def rarefy(df, col, ks):
    """For k randomly chosen reviews (averaged over all k-subsets): expected number of distinct groups (D),
    expected number of groups raised by 2+ of the k reviews (S), and expected share of groups raised by one review (U)."""
    sets = df.groupby("review")[col].agg(set).to_dict()
    reviews = sorted(sets)
    out = {}
    for k in ks:
        if k > len(reviews):
            out[k] = (np.nan, np.nan, np.nan)
            continue
        d, s, u = [], [], []
        for sub in combinations(reviews, k):
            counts = Counter(g for r in sub for g in sets[r])
            d.append(len(counts))
            s.append(sum(c >= 2 for c in counts.values()))
            u.append(np.mean([c == 1 for c in counts.values()]) if counts else np.nan)
        out[k] = (float(np.mean(d)), float(np.mean(s)), float(np.nanmean(u)))
    return out


def pair_rates(df, col):
    """Share of cross-review claim pairs that fall in the same group (Kim et al. style), and mean pairwise hit rate
    (share of one review's claims that have a match in the other, Liang et al. style)."""
    reviews = sorted(df.review.unique())
    pair_hits, pair_total, hits = 0, 0, []
    for a, b in combinations(reviews, 2):
        ga, gb = df[df.review == a][col].to_numpy(), df[df.review == b][col].to_numpy()
        same = ga[:, None] == gb[None, :]
        pair_hits += int(same.sum())
        pair_total += same.size
        hits.append(same.any(axis=1).mean())
        hits.append(same.any(axis=0).mean())
    return (pair_hits / pair_total if pair_total else np.nan, float(np.mean(hits)) if hits else np.nan)


def panel_metrics(df, ks=(1, 2, 3, 4)):
    if df.empty:  # assign() gives no columns when no review in the panel has a claim
        return dict(n_reviews=0, n_claims=0)
    n_reviews = df.review.nunique()
    m = dict(n_reviews=n_reviews, n_claims=len(df))
    reviewers = df.groupby("cluster").review.nunique()
    t_reviewers = df.groupby("target").review.nunique()
    weights = reviewers / reviewers.sum()
    m.update(
        n_clusters=int(reviewers.size), n_targets=int(t_reviewers.size),
        n_shared_clusters=int((reviewers >= 2).sum()), n_shared_targets=int((t_reviewers >= 2).sum()),
        share_shared_claims=float(df.cluster.map(reviewers).ge(2).mean()),
        share_shared_claims_lenient=float(df.target.map(t_reviewers).ge(2).mean()),
        share_unique_clusters=float((reviewers == 1).mean()),
        hill_shannon=float(np.exp(-(weights * np.log(weights)).sum())),
        share_specific=float(df.specific.mean()),
        claims_per_review=len(df) / n_reviews,
    )
    m["pair_rate_strict"], m["hit_rate_strict"] = pair_rates(df, "cluster")
    m["pair_rate_lenient"], m["hit_rate_lenient"] = pair_rates(df, "target")
    for k, (d, sh, u) in rarefy(df, "cluster", ks).items():
        m[f"D{k}"], m[f"S{k}"], m[f"U{k}"] = d, sh, u
    for k, (d, sh, u) in rarefy(df, "target", ks).items():
        m[f"T{k}"], m[f"TS{k}"], m[f"TU{k}"] = d, sh, u
    for c in CATEGORIES:
        m[f"cat_{c}"] = float((df.category == c).mean())
    m.update(_variant(df[df.specific.astype(bool)], "spec"))
    matched = df.sample(frac=1, random_state=0).groupby("review").head(M_MATCH)
    m.update(_variant(matched, f"m{M_MATCH}"))
    return m


def _variant(df, suffix):
    if df.empty:
        return {f"S3_{suffix}": np.nan, f"D3_{suffix}": np.nan, f"share_shared_claims_{suffix}": np.nan}
    reviewers = df.groupby("cluster").review.nunique()
    d, sh, _ = rarefy(df, "cluster", (3,))[3]
    return {f"S3_{suffix}": sh, f"D3_{suffix}": d, f"share_shared_claims_{suffix}": float(df.cluster.map(reviewers).ge(2).mean())}

