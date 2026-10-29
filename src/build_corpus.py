"""Build the review and paper tables from the two Hugging Face dumps.

Outputs data/reviews.parquet and data/papers.parquet. Review text is all free-text fields joined in a fixed order
(summary, main review, strengths, weaknesses, questions, limitations, other), so every form year gives the same unit.
"""

import json
import re
from pathlib import Path

import duckdb
import pandas as pd

DATA = Path("/home/yallen/data/peer-review")
ARCADE = DATA / "arcade/train-*.parquet"
ARCADE_PAPERS = DATA / "arcade/papers.parquet"
ICLR26 = DATA / "iclr2026/heineman.parquet"
ICLR26_RAW = DATA / "iclr2026/jerma_2026.parquet"
OUT = Path(__file__).resolve().parent.parent / "data"

ORDER = [
    ("summary", {"summary", "summary of the paper"}),
    ("main", {"review", "main review", "main_review"}),
    ("strengths", {"strengths", "strength and weaknesses", "strengths_and_weaknesses"}),
    ("weaknesses", {"weaknesses"}),
    ("questions", {"questions"}),
    ("limitations", {"limitations", "limitations_and_societal_impact"}),
    ("other", {"summary of the review", "clarity, quality, novelty and reproducibility", "paper_formatting_concerns"}),
]
WEAK_KEYS = {"weaknesses", "strength and weaknesses", "strengths_and_weaknesses"}
TEXT_KEYS = set().union(*(k for _, k in ORDER))


def unwrap(v):
    return v.get("value") if isinstance(v, dict) else v


def squash(s):
    return re.sub(r"\s+", " ", s or "").strip()


def num(v):
    m = re.match(r"\s*(-?\d+)", str(v)) if v is not None else None
    return int(m.group(1)) if m else None


def join_fields(c):
    parts = []
    for name, keys in ORDER:
        for k in keys:
            if c.get(k):
                parts.append(f"[{name.upper()}] {squash(str(c[k]))}")
    return " ".join(parts)


def decision_class(d):
    d = (d or "").lower()
    if "desk" in d:
        return "desk_reject"
    if "reject" in d or d == "unknown" or d == "":
        return "reject"
    if "withdraw" in d:
        return "withdrawn"
    return "accept"


def from_arcade():
    con = duckdb.connect()
    rows = con.sql(
        f"select venue, review_openreview_id, replyto_openreview_id, writer, content, time "
        f"from read_parquet('{ARCADE}') where title ilike 'Official Review%'"
    ).fetchall()
    out = []
    for venue, rid, pid, writer, content, time in rows:
        try:
            c = {k.lower(): unwrap(v) for k, v in json.loads(content).items()}
        except (json.JSONDecodeError, AttributeError):
            continue
        conf, year = venue.split("/")[0].split(".")[0], int(venue.split("/")[1])
        if conf == "ICLR" and year < 2017 or conf == "EMNLP":
            continue
        weak = next((str(c[k]) for k in c if k in WEAK_KEYS and c[k]), None)
        out.append(dict(
            conf=conf, year=year, paper_id=pid, review_id=rid, reviewer=writer,
            rating=num(c.get("rating") or c.get("recommendation")),
            confidence=num(c.get("confidence")),
            text=join_fields(c), weaknesses=squash(weak) if weak else None,
            form="+".join(sorted(k for k in c if k in TEXT_KEYS)),
            review_time=pd.to_datetime(time, errors="coerce"), modified_time=pd.NaT,
        ))
    reviews = pd.DataFrame(out)
    papers = con.sql(f"select * from '{ARCADE_PAPERS}'").df()
    papers = pd.DataFrame(dict(
        conf=papers.venue.str.split("/").str[0].str.split(".").str[0],
        year=papers.venue.str.split("/").str[1].astype(int),
        paper_id=papers.paper_openreview_id, title=papers.title, abstract=papers.abstract,
        decision_raw=papers.paper_decision, primary_area=None,
    ))
    papers = papers[papers.conf.isin(["ICLR", "NeurIPS"]) & (papers.year >= 2017)]
    return reviews, papers


def from_iclr26():
    con = duckdb.connect()
    df = con.sql(
        f"select r.review_id, r.reviewer, r.rating, r.confidence, r.summary, r.strengths, r.weaknesses, r.review, "
        f"url, venue, title, abstract from (select url, venue, title, abstract, unnest(reviews) r from '{ICLR26}')"
    ).df()
    df["paper_id"] = df.url.str.extract(r"id=([^&]+)")[0]
    raw = con.sql(f"select paper_id, decision, api_raw_review, api_raw_submission from '{ICLR26_RAW}'").df()
    questions, cdate, mdate, area, decision = {}, {}, {}, {}, {}
    for _, r in raw.iterrows():
        try:
            note = json.loads(r.api_raw_review)
        except (TypeError, json.JSONDecodeError):
            continue
        if "id" not in note:
            continue
        content = note.get("content", {})
        questions[note["id"]] = unwrap(content.get("questions")) or ""
        cdate[note["id"]], mdate[note["id"]] = note.get("cdate"), note.get("mdate")
        decision[r.paper_id] = r.decision
        try:
            sub = json.loads(r.api_raw_submission)
            area[r.paper_id] = unwrap(sub.get("content", {}).get("primary_area"))
        except (TypeError, json.JSONDecodeError, AttributeError):
            pass
    df["questions"] = df.review_id.map(questions)
    text = df.apply(lambda r: join_fields(dict(summary=r.summary, strengths=r.strengths, weaknesses=r.weaknesses,
                                               questions=r.questions, review=r.review)), axis=1)
    reviews = pd.DataFrame(dict(
        conf="ICLR", year=2026, paper_id=df.paper_id, review_id=df.review_id, reviewer=df.reviewer,
        rating=df.rating, confidence=df.confidence, text=text, weaknesses=df.weaknesses.map(squash),
        form="questions+strengths+summary+weaknesses",
        review_time=pd.to_datetime(df.review_id.map(cdate), unit="ms", errors="coerce"),
        modified_time=pd.to_datetime(df.review_id.map(mdate), unit="ms", errors="coerce"),
    ))
    p = df.drop_duplicates("paper_id")
    papers = pd.DataFrame(dict(
        conf="ICLR", year=2026, paper_id=p.paper_id, title=p.title, abstract=p.abstract,
        decision_raw=p.venue.where(p.venue.str.contains("Withdrawn"), p.paper_id.map(decision)),
        primary_area=p.paper_id.map(area),
    ))
    return reviews, papers


if __name__ == "__main__":
    r1, p1 = from_arcade()
    r2, p2 = from_iclr26()
    reviews = pd.concat([r1, r2], ignore_index=True)
    reviews = reviews[reviews.text.str.len() > 200].drop_duplicates("review_id")
    papers = pd.concat([p1, p2], ignore_index=True).drop_duplicates("paper_id")
    papers["decision"] = papers.decision_raw.map(decision_class)
    papers["withdrawn"] = papers.decision.eq("withdrawn")
    counts = reviews.groupby("paper_id").size().rename("n_reviews")
    papers = papers.merge(counts, left_on="paper_id", right_index=True, how="left")
    papers["n_reviews"] = papers.n_reviews.fillna(0).astype(int)
    reviews = reviews.merge(papers[["paper_id", "withdrawn", "decision", "primary_area"]], on="paper_id", how="left")
    reviews["withdrawn"] = reviews.withdrawn.fillna(False).astype(bool)
    OUT.mkdir(exist_ok=True)
    reviews.to_parquet(OUT / "reviews.parquet")
    papers.to_parquet(OUT / "papers.parquet")
    summary = reviews[~reviews.withdrawn].groupby(["conf", "year"]).agg(
        papers=("paper_id", "nunique"), reviews=("review_id", "size"),
        median_chars=("text", lambda s: int(s.str.len().median())),
        has_conf=("confidence", lambda s: s.notna().mean().round(2)),
        has_time=("review_time", lambda s: s.notna().mean().round(2)),
    )
    summary["rev_per_paper"] = (summary.reviews / summary.papers).round(2)
    print(summary.to_string())
    print(papers.groupby(["conf", "year"]).decision.value_counts().unstack(fill_value=0).to_string())
