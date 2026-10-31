"""Outcome tables from the pipeline output (pass-1 claims, pass-2 clusters) plus review and paper metadata.

panels: panel-level and review-level outcomes.
    tag "" / "_noq" / "_wd" (ICLR, NeurIPS): data/panel_outcomes_<model><tag>.parquet,
        review_outcomes_<model><tag>.parquet, and placebo_outcomes_ / sub3_outcomes_<model><tag>.parquet when the
        clusters file has such panels.
    tag "_icml" / "_other": data/icml_panel_outcomes_<model>.parquet, data/other_panel_outcomes_<model>.parquet
        (real and placebo panels).
pairs: one row per pair of reviews of the same paper, all sources (real panels only): data/pairs_<model>.parquet with
    panel_id, conf, year, a, b, hit (mean share of one review's strict clusters matched in the other, both directions)
    and hit_spec (same on specific claims). AI flags are joined by the consumers.
standardize: match rate per source-year that does not move with review length and panel size (g-computation), so
    year-to-year changes are not driven by longer reviews or bigger panels. Per source an OLS on pairs,
    hit ~ C(year) + C(size) + cubic splines (4 df) in log chars of the shorter and of the longer review, with
    size = reviews in the panel clipped to the source's smallest size + 2; the rate of year y is the mean prediction
    over all the source's pairs with year set to y. SE: delta method with panel-clustered covariance (exact, the model is linear).
    ICLR starts in 2020, other sources in 2019 (in_window). Writes reports/match_std_<model>.csv (conf, year, pairs, raw, std, std_se in %).
withdrawals: ICLR standardized rate with the withdrawn share equalized across years, rating ICC and match rate by
    rating quintile: reports/match_std_withdrawals.csv.

Usage: outcomes.py panels [--model gpt-6-luna] [--tag _noq]
       outcomes.py pairs [--model gpt-6-luna]
       outcomes.py standardize [--model gpt-6-luna]
       outcomes.py withdrawals [--model gpt-6-luna --target 0.23]
"""

import json
import re
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from extract import CATEGORIES
from metrics import assign, load_jsonl, panel_metrics
from settings import settings

PANEL_FILES = ["panels.jsonl", "panels_icml.jsonl", "panels_other.jsonl"]  # real panels of the pair table
# review metadata per non-ML source tag: file and extra columns copied to the panel table
SOURCES = {"_icml": ("icml_sample.parquet", []), "_other": ("other_reviews.parquet", ["field", "journal"])}
MARKERS = ["commendable", "meticulous", "intricate", "delve", "showcas", "noteworthy", "underscore", "pivotal",
           "seamless", "invaluable", "notably", "realm", "comprehensive", "crucial", "robust", "leverag", "nuanced",
           "multifaceted", "paramount", "elucidat", "garner", "tapestry", "landscape", "holistic", "valuable insights"]
MARKER_RX = re.compile(r"\b(" + "|".join(MARKERS) + ")", re.IGNORECASE)
# Text that marks a review edited after the author response (public reviews are final versions).
EDIT_RX = re.compile(r"(after|post)[- ]?(the )?(rebuttal|discussion|response)|concerns? (have been|were|was|are) (addressed|resolved)"
                     r"|thank(s| you)[^.]{0,40}(response|rebuttal|clarification)|(rais|updat|increas|lower|revis)(e|ed|ing) (my|the) (score|rating|review)"
                     r"|\[?(update|revision|edit)d?[:\]]", re.IGNORECASE)


def load_claims(path):
    # claim text is not needed for outcomes; dropping it keeps the run small (the machine is shared)
    keep = ("category", "target", "specific")
    with open(path) as f:
        rows = map(json.loads, f)
        return {r["review_id"]: [{k: c.get(k) for k in keep} for c in r["claims"]] for r in rows if "claims" in r}


def batches(path, columns, ids, desc):
    """Stream a parquet file in batches, keeping rows whose review_id is in ids (full texts do not fit in memory)."""
    f = pq.ParquetFile(path)
    keep = pa.array(list(ids), type=f.schema_arrow.field("review_id").type)
    for b in tqdm(f.iter_batches(batch_size=20000, columns=columns), total=-(-f.metadata.num_rows // 20000), desc=desc):
        yield b.filter(pc.is_in(b.column("review_id"), value_set=keep))


def review_meta(ids, data=settings.DATA_DIR):
    parts = []
    cols = ["review_id", "paper_id", "conf", "year", "rating", "confidence", "text", "review_time", "modified_time"]
    for b in batches(data / "reviews.parquet", cols, ids, "review text"):
        b = b.to_pandas()
        b["chars"] = b.text.str.len()
        b["n_markers"] = b.text.map(lambda t: len(MARKER_RX.findall(t)))
        b["rebuttal_text"] = b.text.map(lambda t: bool(EDIT_RX.search(t)))
        parts.append(b.drop(columns="text"))
    r = pd.concat(parts, ignore_index=True)
    r["any_marker"] = r.n_markers > 0
    r["marker_rate"] = r.n_markers / (r.chars / 1000)
    # modified_time is a bulk timestamp (99.8% of ICLR 2026 reviews share two dates), so it cannot flag edits.
    # ICLR 2024-2025: a rating that changed after the rebuttal marks an edited review; other years: text markers only.
    r["edited"] = False
    c = pd.read_parquet(data / "review_cdates.parquet", columns=["review_id", "ext_rating_before", "ext_rating_after"])
    c = c.dropna().query("ext_rating_before != ext_rating_after")
    r.loc[r.review_id.isin(c.review_id), "edited"] = True
    return r


def paper_meta(data=settings.DATA_DIR):
    p = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "conf", "year", "decision", "n_reviews"])
    a = pd.read_parquet(data / "paper_areas.parquet", columns=["paper_id", "area", "area_prob"])
    return p.merge(a, on="paper_id", how="left")


def review_rows(df, review_ids, paper):
    reviewers = df.groupby("cluster").review.nunique()
    t_reviewers = df.groupby("target").review.nunique()
    df["shared"] = df.cluster.map(reviewers).ge(2)
    df["shared_lenient"] = df.target.map(t_reviewers).ge(2)
    for j, rid in enumerate(review_ids):
        g = df[df.review == j]
        yield dict(review_id=rid, paper_id=paper, n_claims=len(g),
                   share_shared=g.shared.mean() if len(g) else np.nan,
                   share_shared_lenient=g.shared_lenient.mean() if len(g) else np.nan,
                   n_unique=int((~g.shared).sum()), share_specific=g.specific.mean() if len(g) else np.nan,
                   **{f"cat_{c}": float((g.category == c).mean()) if len(g) else np.nan for c in CATEGORIES})


def ml_panels(model, tag, data, out):
    claims = load_claims(data / f"claims_{model}{tag}.jsonl")
    rmeta = review_meta(claims.keys(), data).set_index("review_id")
    pmeta = paper_meta(data).set_index("paper_id")

    rows, reviews = {"panel": [], "placebo": [], "sub3": []}, []
    for pid, p in tqdm(load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id").items(), desc="panels"):
        # panel ids: <paper_id> (real), noq_ / wd_ / sub3_<paper_id> (variants), placebo(A)_<conf>_<year>_<k> (shuffled)
        kind = "placebo" if pid.startswith("placebo") else "sub3" if pid.startswith("sub3_") else "panel"
        paper = re.sub(r"^(noq|wd|sub3)_", "", pid)
        if kind == "panel" and paper not in pmeta.index:
            continue  # blind-set panels (B00...) share the clusters file
        df = assign(p, claims)
        m = panel_metrics(df)
        rv = rmeta.reindex(p["review_ids"])
        m.update(panel_id=pid, total_chars=float(rv.chars.sum()), rating_mean=rv.rating.mean(),
                 rating_sd=rv.rating.std(ddof=0), confidence_mean=rv.confidence.mean(),
                 n_marker_reviews=int(rv.any_marker.fillna(False).sum()),
                 n_rebuttal_text=int(rv.rebuttal_text.fillna(False).sum()), n_edited=int(rv.edited.fillna(False).sum()))
        if kind == "placebo":
            _, conf, year, _ = pid.split("_")
            m.update(conf=conf, year=int(year), placebo_kind="same_area3" if pid.startswith("placeboA_") else "shuffled4")
        else:
            m.update(pmeta.loc[paper].to_dict(), paper_id=paper)
        rows[kind].append(m)
        if kind == "panel" and not df.empty:
            reviews.extend(review_rows(df, p["review_ids"], paper))
    for kind, prow in rows.items():
        if prow:
            pd.DataFrame(prow).to_parquet(out / f"{kind}_outcomes_{model}{tag}.parquet")
            print(kind, len(prow), "panels")
    if rows["panel"]:
        R = pd.DataFrame(reviews).merge(rmeta.drop(columns=["paper_id"]), left_on="review_id", right_index=True, how="left")
        R = R.merge(pmeta[["decision", "area"]], left_on="paper_id", right_index=True, how="left")
        R.to_parquet(out / f"review_outcomes_{model}{tag}.parquet")
        print("reviews", len(R))


def source_panels(model, tag, data, out):
    meta_file, extra = SOURCES[tag]
    claims = load_claims(data / f"claims_{model}{tag}.jsonl")
    meta = pd.read_parquet(data / meta_file, columns=["review_id", "conf", "year", "rating"] + extra).set_index("review_id")
    rows = []
    for pid, p in tqdm(load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id").items(), desc=tag[1:]):
        df = assign(p, claims)
        if df.empty:
            continue
        m = panel_metrics(df, ks=(2,))
        first = meta.loc[p["review_ids"][0]]
        m.update(panel_id=pid, placebo=pid.startswith("otherplacebo_"), conf=first.conf, year=int(first.year),
                 rating_sd=meta.rating.reindex(p["review_ids"]).std(ddof=0), **first[extra].to_dict())
        rows.append(m)
    out_table = pd.DataFrame(rows)
    out_table.to_parquet(out / f"{tag[1:]}_panel_outcomes_{model}.parquet")
    print(out_table.groupby(["placebo", "conf", "year"]).hit_rate_strict.mean().mul(100).round(2).unstack(0).to_string())


def panels(model=settings.DEFAULT_MODEL, tag="", data=str(settings.DATA_DIR), out=str(settings.DATA_DIR)):
    (source_panels if tag in SOURCES else ml_panels)(model, tag, Path(data), Path(out))


def hit(x, y):
    same = x[:, None] == y[None, :]
    return (same.any(1).mean() + same.any(0).mean()) / 2


def panel_pairs(df, review_ids):
    for a, b in combinations(range(len(review_ids)), 2):
        ga, gb = df[df.review == a], df[df.review == b]
        if ga.empty or gb.empty:
            continue
        sa, sb = ga[ga.specific.astype(bool)], gb[gb.specific.astype(bool)]
        yield dict(a=review_ids[a], b=review_ids[b], hit=hit(ga.cluster.to_numpy(), gb.cluster.to_numpy()),
                   hit_spec=hit(sa.cluster.to_numpy(), sb.cluster.to_numpy()) if len(sa) and len(sb) else np.nan)


def pairs(model=settings.DEFAULT_MODEL, data=str(settings.DATA_DIR), out=str(settings.DATA_DIR)):
    data = Path(data)
    papers = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    rows = []
    for tag in ["", *SOURCES]:
        if not (data / f"clusters_{model}{tag}.jsonl").exists():
            continue
        claims = load_claims(data / f"claims_{model}{tag}.jsonl")
        if tag:
            meta = pd.read_parquet(data / SOURCES[tag][0], columns=["review_id", "conf", "year"]).set_index("review_id")
        for pid, p in tqdm(load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id").items(), desc=f"pairs{tag}"):
            if tag:
                if not pid.startswith(("icml_", "other_")):
                    continue
                conf, year = meta.loc[p["review_ids"][0], ["conf", "year"]]
            else:
                if pid not in papers.index:
                    continue
                conf, year = papers.loc[pid, ["conf", "year"]]
            df = assign(p, claims)
            if df.empty:
                continue
            rows.extend(dict(panel_id=pid, conf=conf, year=int(year), **r) for r in panel_pairs(df, p["review_ids"]))
    T = pd.DataFrame(rows)
    T.to_parquet(Path(out) / f"pairs_{model}.parquet")
    print(T.groupby(["conf", "year"]).agg(pairs=("hit", "size"), hit=("hit", "mean")).to_string())


def review_chars(data, ids):
    parts = []
    for name in ["reviews.parquet", "icml_sample.parquet", "other_reviews.parquet"]:
        for b in batches(data / name, ["review_id", "text"], ids, name):
            parts.append(pd.Series(pc.utf8_length(b.column("text")).to_numpy(), index=b.column("review_id").to_pandas()))
    L = pd.concat(parts)
    return L[~L.index.duplicated()]


# Before 2020 the public ICLR data hold almost no withdrawn papers (later 13-26%), and withdrawn papers' reviews match
# more, so earlier ICLR years are not comparable (see `withdrawals`). Every other source starts in 2019.
ICLR_FROM = 2020
FROM_YEAR = {"ICLR": ICLR_FROM}
DEFAULT_FROM = 2019


def in_window(df):
    return df[df.year >= df.conf.map(FROM_YEAR).fillna(DEFAULT_FROM)]


def pair_design(model, data, all_years=False):
    P = pd.read_parquet(data / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    if not all_years:
        P = in_window(P)
    L = review_chars(data, set(P.a) | set(P.b))
    la, lb = np.log(L.reindex(P.a).to_numpy()), np.log(L.reindex(P.b).to_numpy())
    P["log_short"], P["log_long"] = np.minimum(la, lb), np.maximum(la, lb)
    sizes = {}
    for name in PANEL_FILES:
        sizes.update((p["panel_id"], len(p["review_ids"])) for p in map(json.loads, open(data / name)))
    P["size"] = P.panel_id.map(sizes)
    P["size"] = P["size"].clip(upper=P.groupby("conf")["size"].transform("min") + 2)
    return P


STD_FORMULA = "hit ~ C(year) + C(size) + bs(log_short, df=4) + bs(log_long, df=4)"


def year_rates(S, weights=None):
    """Standardized rate per year for one source's pairs S: {year: (std, std_se)} in %."""
    import statsmodels.formula.api as smf

    # splines: in ICLR both short and long reviews match more than mid-length ones
    model = smf.ols(STD_FORMULA, data=S) if weights is None else smf.wls(STD_FORMULA, data=S, weights=weights)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(S.panel_id)[0]})
    names, V = pd.Index(fit.model.exog_names), fit.cov_params().to_numpy()
    is_year = names.str.startswith("C(year)")
    x_mean = np.average(fit.model.exog, axis=0, weights=weights)
    out = {}
    for year in sorted(S.year.unique()):
        # mean design row of all the source's pairs with year set to `year`
        x = np.where(is_year, names == f"C(year)[T.{year}]", x_mean)
        out[year] = (x @ fit.params.to_numpy() * 100, np.sqrt(x @ V @ x) * 100)
    return out


def standardize(model=settings.DEFAULT_MODEL, data=str(settings.DATA_DIR), out=None):
    out = out or settings.REPORTS_DIR / f"match_std_{model}.csv"
    P = pair_design(model, Path(data))
    rows = []
    for conf, S in tqdm(P.groupby("conf"), desc="sources"):
        for year, (std, se) in year_rates(S).items():
            g = S[S.year == year]
            rows.append(dict(conf=conf, year=year, pairs=len(g), raw=g.hit.mean() * 100, std=std, std_se=se))
    T = pd.DataFrame(rows)
    T.to_csv(out, index=False)
    print(T.round(2).to_string(index=False))


# Withdrawn ICLR submissions per year (OpenReview counts as tabulated by reviewer3.com, accessed 2026-10-30). The source
# dump has no reviews of withdrawn papers before 2026; our 2026 data has them and the pairs drop them.
ICLR_SUBMITTED = {2017: 490, 2018: 1018, 2019: 1579, 2020: 2594, 2021: 3014, 2022: 3422, 2023: 4955, 2024: 7404,
                  2025: 11672}
ICLR_WITHDRAWN = {2017: 0, 2018: 83, 2019: 160, 2020: 369, 2021: 403, 2022: 779, 2023: 1145, 2024: 1661, 2025: 2991}


def withdrawals(model=settings.DEFAULT_MODEL, target=0.23, data=str(settings.DATA_DIR), out=str(settings.REPORTS_DIR / "match_std_withdrawals.csv")):
    """ICLR standardized rate with the share of withdrawn submissions equalized across years.

    Withdrawn papers are mostly the weakest, and their reviews match more, so years whose data still hold them (2017-2019)
    look higher. Each year y with withdrawn share w_y < target loses a further (target - w_y) / (1 - w_y) of its papers:
    "drop" removes the lowest-rated papers (an upper bound on the correction); "weight" downweights papers by the ICLR 2026
    withdrawal rate of their rating decile, scaled to the same share. Also writes rating ICC and the match rate by rating
    quintile per year, both free of the claim pipeline's selection of papers.
    """
    data = Path(data)
    R = pd.read_parquet(data / "reviews.parquet", columns=["paper_id", "review_id", "rating", "year", "withdrawn"],
                        filters=[("conf", "==", "ICLR")])
    m = R.groupby("paper_id").agg(year=("year", "first"), rating=("rating", "mean"), withdrawn=("withdrawn", "first"))
    m["pct"] = m.groupby("year").rating.rank(pct=True)
    decile = np.minimum((m.pct * 10).astype(int), 9)
    late = m.year == 2026
    profile = m[late].groupby(decile[late]).withdrawn.mean()
    share = {y: ICLR_WITHDRAWN[y] / ICLR_SUBMITTED[y] for y in ICLR_SUBMITTED}
    m["extra"] = [max(0.0, (target - share[y]) / (1 - share[y])) if y in share else 0.0 for y in m.year]
    m["p"] = profile.reindex(decile).to_numpy()
    m["weight"] = 1 - np.clip(m.p * m.extra / m.groupby("year").p.transform("mean"), 0, 1)

    P = pair_design(model, data, all_years=True)
    P = P[P.conf == "ICLR"].copy()
    paper = R.set_index("review_id").paper_id.reindex(P.a).to_numpy()
    P["pct"], P["weight"] = m.pct.reindex(paper).to_numpy(), m.weight.reindex(paper).to_numpy()
    P["keep"] = P.pct > m.extra.reindex(paper).to_numpy()
    rows = {"as published": year_rates(P), "weight": year_rates(P, P.weight.to_numpy()),
            "drop": year_rates(P[P.keep])}
    T = pd.DataFrame({k: {y: v[0] for y, v in r.items()} for k, r in rows.items()})
    T = T.join(pd.DataFrame({f"{k}_se": {y: v[1] for y, v in r.items()} for k, r in rows.items()}))
    T["withdrawn_share"] = pd.Series(share)
    T["extra_drop"] = m.groupby("year").extra.first()

    def icc(g):
        z = (g.rating - g.rating.mean()) / g.rating.std()
        return 1 - z.groupby(g.paper_id).var().mean() / z.var()

    rated = R[~R.withdrawn & R.rating.notna()]
    T["rating_icc"] = rated.groupby("year")[["paper_id", "rating"]].apply(icc)
    pm = P.groupby(["year", "panel_id"]).agg(hit=("hit", "mean"), pct=("pct", "first")).reset_index()
    q = pm.groupby(["year", pd.cut(pm.pct, [0, .2, .4, .6, .8, 1], labels=[f"q{i}" for i in range(1, 6)])],
                   observed=True).hit.mean().unstack().mul(100)
    T = T.join(q.add_prefix("raw_rating_"))
    T.index.name = "year"
    T.to_csv(out)
    print(T.round(2).to_string())


if __name__ == "__main__":
    fire.Fire({"panels": panels, "pairs": pairs, "standardize": standardize, "withdrawals": withdrawals})
