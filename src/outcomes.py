"""Build panel-level and review-level outcome tables from the pipeline output plus paper metadata.

Usage: outcomes.py <model> [tag]
Writes data/panel_outcomes_<model><tag>.parquet, data/placebo_outcomes_<model><tag>.parquet (if present) and
data/review_outcomes_<model><tag>.parquet.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from extract import CATEGORIES
from metrics import assign, load_jsonl, panel_metrics

DATA = Path(__file__).resolve().parent.parent / "data"
MARKERS = ["commendable", "meticulous", "intricate", "delve", "showcas", "noteworthy", "underscore", "pivotal",
           "seamless", "invaluable", "notably", "realm", "comprehensive", "crucial", "robust", "leverag", "nuanced",
           "multifaceted", "paramount", "elucidat", "garner", "tapestry", "landscape", "holistic", "valuable insights"]
MARKER_RX = re.compile(r"\b(" + "|".join(MARKERS) + ")", re.I)
# Text that marks a review edited after the author response (public reviews are final versions).
EDIT_RX = re.compile(r"(after|post)[- ]?(the )?(rebuttal|discussion|response)|concerns? (have been|were|was|are) (addressed|resolved)"
                     r"|thank(s| you)[^.]{0,40}(response|rebuttal|clarification)|(rais|updat|increas|lower|revis)(e|ed|ing) (my|the) (score|rating|review)"
                     r"|\[?(update|revision|edit)d?[:\]]", re.I)


def review_meta():
    r = pd.read_parquet(DATA / "reviews.parquet",
                        columns=["review_id", "paper_id", "conf", "year", "rating", "confidence", "text",
                                 "review_time", "modified_time"])
    r["chars"] = r.text.str.len()
    r["n_markers"] = r.text.map(lambda t: len(MARKER_RX.findall(t)))
    r["any_marker"] = r.n_markers > 0
    r["marker_rate"] = r.n_markers / (r.chars / 1000)
    # modified_time is a bulk timestamp (99.8% of ICLR 2026 reviews share two dates), so it cannot flag edits.
    # ICLR 2024-2025: a rating that changed after the rebuttal marks an edited review; other years: text markers only.
    r["edited"] = False
    c = pd.read_parquet(DATA / "review_cdates.parquet", columns=["review_id", "ext_rating_before", "ext_rating_after"])
    c = c.dropna().query("ext_rating_before != ext_rating_after")
    r.loc[r.review_id.isin(c.review_id), "edited"] = True
    r["rebuttal_text"] = r.text.map(lambda t: bool(EDIT_RX.search(t)))
    return r.drop(columns="text")


def paper_meta():
    p = pd.read_parquet(DATA / "papers.parquet", columns=["paper_id", "conf", "year", "decision", "n_reviews"])
    a = pd.read_parquet(DATA / "paper_areas.parquet", columns=["paper_id", "area", "area_prob"])
    return p.merge(a, on="paper_id", how="left")


def build(model, tag=""):
    claims = {k: v["claims"] for k, v in load_jsonl(DATA / f"claims_{model}{tag}.jsonl", "review_id").items()}
    rmeta = review_meta().set_index("review_id")
    pmeta = paper_meta().set_index("paper_id")

    panels = load_jsonl(DATA / f"clusters_{model}{tag}.jsonl", "panel_id")
    # panel ids: <paper_id> (real panel), sub3_<paper_id> / noq_<paper_id> (variants), placebo_* / placeboA_* (shuffled)
    groups = {"panel": {}, "placebo": {}, "sub3": {}}
    for pid, p in panels.items():
        if pid.startswith("placebo"):
            groups["placebo"][pid] = p
        elif pid.startswith("sub3_"):
            groups["sub3"][pid] = p
        else:
            paper = pid.removeprefix("noq_").removeprefix("wd_")
            if paper in pmeta.index:
                groups["panel"][pid] = p
    for kind, panels in groups.items():
        if not panels:
            continue
        prow, rrow = [], []
        for pid, p in panels.items():
            paper = pid.removeprefix("noq_").removeprefix("sub3_").removeprefix("wd_")
            df = assign(p, claims)
            m = panel_metrics(df)
            rv = rmeta.reindex(p["review_ids"])
            m.update(panel_id=pid, total_chars=float(rv.chars.sum()), rating_mean=rv.rating.mean(),
                     rating_sd=rv.rating.std(ddof=0), confidence_mean=rv.confidence.mean(),
                     n_marker_reviews=int(rv.any_marker.fillna(False).sum()),
                     n_rebuttal_text=int(rv.rebuttal_text.fillna(False).sum()), n_edited=int(rv.edited.fillna(False).sum()))
            if kind == "placebo":
                m.update(conf=rv.conf.iloc[0], year=int(rv.year.iloc[0]), placebo_kind="same_area3" if pid.startswith("placeboA_") else "shuffled4")
            else:
                m.update(pmeta.loc[paper].to_dict(), paper_id=paper)
            prow.append(m)
            if kind == "panel" and not df.empty:
                reviewers = df.groupby("cluster").review.nunique()
                t_reviewers = df.groupby("target").review.nunique()
                df["shared"] = df.cluster.map(reviewers).ge(2)
                df["shared_lenient"] = df.target.map(t_reviewers).ge(2)
                for j, rid in enumerate(p["review_ids"]):
                    g = df[df.review == j]
                    row = dict(review_id=rid, paper_id=paper, n_claims=len(g),
                               share_shared=g.shared.mean() if len(g) else np.nan,
                               share_shared_lenient=g.shared_lenient.mean() if len(g) else np.nan,
                               n_unique=int((~g.shared).sum()), share_specific=g.specific.mean() if len(g) else np.nan,
                               **{f"cat_{c}": float((g.category == c).mean()) if len(g) else np.nan for c in CATEGORIES})
                    rrow.append(row)
        out = pd.DataFrame(prow)
        out.to_parquet(DATA / f"{kind}_outcomes_{model}{tag}.parquet")
        print(kind, len(out), "panels")
        if kind == "panel":
            R = pd.DataFrame(rrow).merge(rmeta.drop(columns=["paper_id"]), left_on="review_id", right_index=True, how="left")
            R = R.merge(pmeta[["decision", "area"]], left_on="paper_id", right_index=True, how="left")
            R.to_parquet(DATA / f"review_outcomes_{model}{tag}.parquet")
            print("reviews", len(R))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
