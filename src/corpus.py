"""Build the review and paper tables.

ml: ICLR/NeurIPS reviews from the two Hugging Face dumps (ResearchArcade; davidheineman/iclr-2026 with
JerMa88/ICLR_Peer_Reviews_2026 for questions, dates and areas) -> data/reviews.parquet, data/papers.parquet.
Review text is all free-text fields joined in a fixed order (summary, main review, strengths, weaknesses, questions,
limitations, other), so every form year gives the same unit.

other: merges the per-source outputs of `fetch.py` into data/other_reviews.parquet and data/other_papers.parquet, runs
the quality checks and writes reports/other_inventory.txt. Reviews: conf, field, journal, year, paper_id, review_id,
reviewer, rating, rating_raw, text, review_time, n_chars. The rating scale differs by source (see fetch.py); compare
ratings only within a conf.

areas: primary-area label for every sampled paper. ICLR 2026 submissions carry an OpenReview primary_area. We embed
title + abstract (text-embedding-3-small), fit a logistic regression on the 2026 labels, and predict the area of
earlier papers. Writes data/abstract_emb.npy and data/paper_areas.parquet (paper_id, area, area_prob, area_pred, row =
embedding row index). Calls the OpenAI API (key from .env via settings.py).

Usage: corpus.py ml [--out data]
       corpus.py other [--src data/other] [--out data] [--report reports/other_inventory.txt]
       corpus.py areas [--out data]
"""

import sys
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import settings

ARCADE = settings.RAW_ML_REVIEWS_DIR / "arcade/train-*.parquet"
ARCADE_PAPERS = settings.RAW_ML_REVIEWS_DIR / "arcade/papers.parquet"
ICLR26 = settings.RAW_ML_REVIEWS_DIR / "iclr2026/heineman.parquet"
ICLR26_RAW = settings.RAW_ML_REVIEWS_DIR / "iclr2026/jerma_2026.parquet"
REPORT = settings.REPORTS_DIR / "other_inventory.txt"

# tuples, not sets: ICLR 2023 fills two "other" keys, and set order changes with the hash seed of each run
ORDER = [
    ("summary", ("summary", "summary of the paper")),
    ("main", ("review", "main review", "main_review")),
    ("strengths", ("strengths", "strength and weaknesses", "strengths_and_weaknesses")),
    ("weaknesses", ("weaknesses",)),
    ("questions", ("questions",)),
    ("limitations", ("limitations", "limitations_and_societal_impact")),
    ("other", ("summary of the review", "clarity, quality, novelty and reproducibility", "paper_formatting_concerns")),
]
WEAK_KEYS = {"weaknesses", "strength and weaknesses", "strengths_and_weaknesses"}
TEXT_KEYS = set().union(*(k for _, k in ORDER))

SOURCES = ["scipost", "f1000", "bmc", "plos", "copernicus"]
REVIEW_COLS = ["conf", "field", "journal", "year", "paper_id", "review_id", "reviewer", "rating", "rating_raw", "text",
               "review_time", "n_chars"]
PAPER_COLS = ["conf", "field", "journal", "year", "paper_id", "title", "n_reviews", "licence"]


def unwrap(v):
    return v.get("value") if isinstance(v, dict) else v


def squash(s):
    return re.sub(r"\s+", " ", s or "").strip()


def num(v):
    m = re.match(r"\s*(-?\d+)", str(v)) if v is not None else None
    return int(m.group(1)) if m else None


def join_fields(c):
    return " ".join(f"[{name.upper()}] {squash(str(c[k]))}" for name, keys in ORDER for k in keys if c.get(k))


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
    for venue, rid, pid, writer, content, when in tqdm(rows, desc="arcade reviews"):
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
            review_time=pd.to_datetime(when, errors="coerce"), modified_time=pd.NaT,
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
    for _, r in tqdm(raw.iterrows(), total=len(raw), desc="iclr2026 raw notes"):
        decision[r.paper_id] = r.decision
        try:
            sub = json.loads(r.api_raw_submission)
            area[r.paper_id] = unwrap(sub.get("content", {}).get("primary_area"))
        except (TypeError, json.JSONDecodeError, AttributeError):
            pass
        try:
            note = json.loads(r.api_raw_review)
        except (TypeError, json.JSONDecodeError):
            continue
        if "id" not in note:
            continue
        questions[note["id"]] = unwrap(note.get("content", {}).get("questions")) or ""
        cdate[note["id"]], mdate[note["id"]] = note.get("cdate"), note.get("mdate")
    # NaN is truthy, so an unmatched review would get "[QUESTIONS] nan"
    df["questions"] = df.review_id.map(questions).fillna("")
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


def merge_ml(parts):
    """[(reviews, papers)] -> deduplicated reviews with paper decisions, papers with review counts."""
    reviews = pd.concat([r for r, _ in parts], ignore_index=True)
    reviews = reviews[reviews.text.str.len() > 200].drop_duplicates("review_id")
    # The dump repeats one review's text under another note of the same paper (25% of ICLR 2017, 8% of 2019).
    reviews = reviews.drop_duplicates(["paper_id", "text"])
    papers = pd.concat([p for _, p in parts], ignore_index=True).drop_duplicates("paper_id")
    papers["decision"] = papers.decision_raw.map(decision_class)
    papers["withdrawn"] = papers.decision.eq("withdrawn")
    counts = reviews.groupby("paper_id").size().rename("n_reviews")
    papers = papers.merge(counts, left_on="paper_id", right_index=True, how="left")
    papers["n_reviews"] = papers.n_reviews.fillna(0).astype(int)
    reviews = reviews.merge(papers[["paper_id", "withdrawn", "decision", "primary_area"]], on="paper_id", how="left")
    reviews["withdrawn"] = reviews.withdrawn.fillna(False).astype(bool)
    return reviews, papers


def ml(out=settings.DATA_DIR):
    out = Path(out)
    reviews, papers = merge_ml([from_arcade(), from_iclr26()])
    out.mkdir(exist_ok=True)
    reviews.to_parquet(out / "reviews.parquet")
    papers.to_parquet(out / "papers.parquet")
    summary = reviews[~reviews.withdrawn].groupby(["conf", "year"]).agg(
        papers=("paper_id", "nunique"), reviews=("review_id", "size"),
        median_chars=("text", lambda s: int(s.str.len().median())),
        has_conf=("confidence", lambda s: s.notna().mean().round(2)),
        has_time=("review_time", lambda s: s.notna().mean().round(2)),
    )
    summary["rev_per_paper"] = (summary.reviews / summary.papers).round(2)
    print(summary.to_string())
    print(papers.groupby(["conf", "year"]).decision.value_counts().unstack(fill_value=0).to_string())


def inventory(rv, pp, have, dups, src):
    out = [f"data/other_reviews.parquet: {len(rv)} reviews, {rv.paper_id.nunique()} papers; sources: {', '.join(have)}", ""]
    inv = rv.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size"),
                                           median_chars=("n_chars", "median"), share_rated=("rating", lambda r: r.notna().mean()))
    inv.insert(2, "rev_per_paper", inv.reviews / inv.papers)
    out += ["Per source x year", inv.round(2).to_string(), ""]
    out += ["Totals by source", rv.groupby("conf").agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string(), ""]
    out += [f"Exact-duplicate texts within a paper in the merged file: {dups} (dropped)"]
    for s in have:
        f = src / f"{s}_stats.json"
        if f.exists():
            out.append(f"  {s} fetch stats: {json.loads(f.read_text())}")
    if (src / "plos_stats.json").exists():
        st = json.loads((src / "plos_stats.json").read_text())
        out.append(f"PLOS: letters where reviewer splitting found 2+ reviewers: {st['letters_split_2plus']}/{st['letters']}"
                   f" = {st['letters_split_2plus'] / st['letters']:.1%}")
    out += ["", "Licences (papers)", pp.groupby(["conf", "licence"], dropna=False).size().to_string(), ""]
    out += ["Journals", rv.groupby(["conf", "journal"]).paper_id.nunique().sort_values(ascending=False).groupby(level=0).head(6).to_string(), ""]
    out.append("Random snippets (first 300 chars)")
    for conf, g in rv.groupby("conf"):
        for r in g.sample(3, random_state=0).itertuples():
            out.append(f"[{conf} {r.year} {r.paper_id} {r.reviewer} rating={r.rating_raw}] {r.text[:300]}")
        out.append("")
    return "\n".join(out)


def other(src=settings.DATA_DIR / "other", out=settings.DATA_DIR, report=REPORT):
    src, out = Path(src), Path(out)
    have = [s for s in SOURCES if (src / f"{s}_reviews.parquet").exists()]
    rv = pd.concat([pd.read_parquet(src / f"{s}_reviews.parquet") for s in have], ignore_index=True)
    pp = pd.concat([pd.read_parquet(src / f"{s}_papers.parquet") for s in have], ignore_index=True)
    for df in (rv, pp):  # Europe PMC spells PLOS journal titles three ways
        df["journal"] = df.journal.str.replace(r"^PLoS\b", "PLOS", regex=True).replace({"PLOS One": "PLOS ONE"})
    rv["rating"] = pd.to_numeric(rv.rating).round().astype("Int64")
    rv["rating_raw"] = rv.rating_raw.astype("string")
    rv["year"] = rv.year.astype(int)
    rv = rv[REVIEW_COLS]
    dups = int(rv.duplicated(["paper_id", "text"]).sum())
    rv = rv.drop_duplicates(["paper_id", "text"])
    assert rv.review_id.is_unique
    n = rv.groupby("paper_id").size()
    rv = rv[rv.paper_id.isin(n[n >= 2].index)].reset_index(drop=True)
    pp = pp[pp.paper_id.isin(rv.paper_id)].copy()
    pp["n_reviews"] = pp.paper_id.map(rv.groupby("paper_id").size())
    pp = pp[PAPER_COLS].reset_index(drop=True)
    rv.to_parquet(out / "other_reviews.parquet")
    pp.to_parquet(out / "other_papers.parquet")
    text = inventory(rv, pp, have, dups, src)
    print(text)
    Path(report).write_text(text + "\n")


def embed(docs, chunk=256):
    from openai import OpenAI, RateLimitError  # slow import, only needed here

    client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2)

    def one(part):
        while True:
            try:
                r = client.embeddings.create(model="text-embedding-3-small", input=part)
                return [e.embedding for e in r.data]
            except RateLimitError:
                time.sleep(10)

    chunks = [docs[i:i + chunk] for i in range(0, len(docs), chunk)]
    with ThreadPoolExecutor(2) as pool:
        parts = tqdm(pool.map(one, chunks), total=len(chunks), desc="embed abstracts")
        return np.array([e for part in parts for e in part], dtype=np.float32)


def areas(out=settings.DATA_DIR):
    from sklearn.linear_model import LogisticRegression  # slow import, only needed here
    from sklearn.model_selection import cross_val_score

    out = Path(out)
    papers = pd.read_parquet(settings.DATA_DIR / "papers.parquet")
    sampled = {json.loads(line)["panel_id"] for line in (settings.DATA_DIR / "panels.jsonl").read_text().splitlines()}
    need = papers[papers.paper_id.isin(sampled) | ((papers.year == 2026) & papers.primary_area.notna())].copy()
    need["doc"] = (need.title.fillna("") + "\n" + need.abstract.fillna("")).str.slice(0, 6000)
    need = need[need.doc.str.len() > 50].reset_index(drop=True)
    print(len(need), "papers to embed")
    emb = embed(need.doc.tolist())
    np.save(out / "abstract_emb.npy", emb)

    labelled = need.primary_area.notna().to_numpy()
    y = need.primary_area[labelled].to_numpy()
    clf = LogisticRegression(max_iter=2000, C=3.0)
    print("5-fold accuracy on 2026 labels:", cross_val_score(clf, emb[labelled], y, cv=5).mean().round(3))
    clf.fit(emb[labelled], y)
    proba = clf.predict_proba(emb)
    pred = clf.classes_[proba.argmax(1)]
    res = pd.DataFrame(dict(paper_id=need.paper_id, area=np.where(labelled, need.primary_area, pred),
                            area_prob=proba.max(1), area_pred=pred, row=np.arange(len(need))))
    res.to_parquet(out / "paper_areas.parquet")
    print(res.merge(papers[["paper_id", "conf", "year"]]).groupby(["conf", "year"]).area.value_counts(normalize=True)
          .unstack(fill_value=0).round(2).T.to_string())


if __name__ == "__main__":
    fire.Fire(dict(ml=ml, other=other, areas=areas))
