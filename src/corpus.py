import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import duckdb
import fire
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import settings

ARCADE = settings.RAW_ML_REVIEWS_DIR / "arcade/train-*.parquet"
ARCADE_PAPERS = settings.RAW_ML_REVIEWS_DIR / "arcade/papers.parquet"
ICLR26 = settings.RAW_ML_REVIEWS_DIR / "iclr2026/heineman.parquet"
ICLR26_RAW = settings.RAW_ML_REVIEWS_DIR / "iclr2026/jerma_2026.parquet"
REPORT = settings.REPORTS_DIR / "other_inventory.txt"

SECTION_KEYS = [
    ("summary", ("summary", "summary of the paper")),
    ("main", ("review", "main review", "main_review")),
    ("strengths", ("strengths", "strength and weaknesses", "strengths_and_weaknesses")),
    ("weaknesses", ("weaknesses",)),
    ("questions", ("questions",)),
    ("limitations", ("limitations", "limitations_and_societal_impact")),
    ("other", ("summary of the review", "clarity, quality, novelty and reproducibility", "paper_formatting_concerns")),
]
WEAK_KEYS = {"weaknesses", "strength and weaknesses", "strengths_and_weaknesses"}
TEXT_KEYS = set().union(*(keys for _, keys in SECTION_KEYS))
MIN_REVIEW_CHARS = 200
NEAR_COPY_CHARS = 3000
NEAR_COPY_RATIO = 0.8

SOURCES = ["scipost", "f1000", "bmc", "plos", "copernicus"]
REVIEW_COLS = [
    "conf",
    "field",
    "journal",
    "year",
    "paper_id",
    "review_id",
    "reviewer",
    "rating",
    "rating_raw",
    "text",
    "review_time",
    "n_chars",
]
PAPER_COLS = ["conf", "field", "journal", "year", "paper_id", "title", "n_reviews", "licence"]
MIN_REVIEWS_PER_PAPER = 2
SNIPPETS_PER_SOURCE = 3
SNIPPET_CHARS = 300
TOP_JOURNALS_PER_SOURCE = 6


def unwrap(v):
    return v.get("value") if isinstance(v, dict) else v


def squash(s):
    return re.sub(r"\s+", " ", s).strip()


def leading_int(value):
    if value is None:
        return None
    match = re.match(r"\s*(-?\d+)", str(value))
    return int(match.group(1)) if match else None


def join_fields(fields):
    parts = []
    for section, keys in SECTION_KEYS:
        for key in keys:
            if fields.get(key):
                parts.append(f"[{section.upper()}] {squash(str(fields[key]))}")
    return " ".join(parts)


def decision_class(decision):
    decision = decision.lower()
    if "desk" in decision:
        return "desk_reject"
    if "reject" in decision or decision in ("", "unknown"):
        return "reject"
    if "withdraw" in decision:
        return "withdrawn"
    if "workshop" in decision:
        return "reject"
    return "accept"


def from_arcade():
    con = duckdb.connect()
    rows = con.sql(
        f"select venue, review_openreview_id, replyto_openreview_id, writer, content, time "
        f"from read_parquet('{ARCADE}') where title ilike 'Official Review%'"
    ).fetchall()
    review_rows = []
    for venue, review_id, paper_id, writer, content, when in tqdm(rows, desc="arcade reviews"):
        try:
            fields = {k.lower(): unwrap(v) for k, v in json.loads(content).items()}
        except json.JSONDecodeError, AttributeError:
            continue
        venue_parts = venue.split("/")
        conf = venue_parts[0].split(".")[0]
        year = int(venue_parts[1])
        if conf == "EMNLP" or (conf == "ICLR" and year < 2017):
            continue
        weaknesses = next((str(fields[k]) for k in fields if k in WEAK_KEYS and fields[k]), None)
        review_rows.append(
            dict(
                conf=conf,
                year=year,
                paper_id=paper_id,
                review_id=review_id,
                reviewer=writer,
                rating=leading_int(fields.get("rating") or fields.get("recommendation")),
                confidence=leading_int(fields.get("confidence")),
                text=join_fields(fields),
                weaknesses=squash(weaknesses) if weaknesses else None,
                form="+".join(sorted(k for k in fields if k in TEXT_KEYS)),
                review_time=pd.to_datetime(when, errors="coerce"),
                modified_time=pd.NaT,
            )
        )
    reviews = pd.DataFrame(review_rows)
    raw_papers = con.sql(f"select * from '{ARCADE_PAPERS}'").df()
    paper_venue_parts = raw_papers.venue.str.split("/")
    papers = pd.DataFrame(
        dict(
            conf=paper_venue_parts.str[0].str.split(".").str[0],
            year=paper_venue_parts.str[1].astype(int),
            paper_id=raw_papers.paper_openreview_id,
            title=raw_papers.title,
            abstract=raw_papers.abstract,
            decision_raw=raw_papers.paper_decision,
            primary_area=None,
        )
    )
    papers = papers[papers.conf.isin(["ICLR", "NeurIPS"]) & (papers.year >= 2017)]
    return reviews, papers


def from_iclr26():
    con = duckdb.connect()
    snapshot_reviews = con.sql(
        f"select r.review_id, r.reviewer, r.rating, r.confidence, r.summary, r.strengths, r.weaknesses, r.review, "
        f"url, venue, title, abstract from (select url, venue, title, abstract, unnest(reviews) r from '{ICLR26}')"
    ).df()
    snapshot_reviews["paper_id"] = snapshot_reviews.url.str.extract(r"id=([^&]+)")[0]
    raw_notes = con.sql(f"select paper_id, decision, api_raw_review, api_raw_submission from '{ICLR26_RAW}'").df()
    decision = {}
    area = {}
    final_venue = {}
    questions = {}
    created = {}
    modified = {}
    for _, raw_row in tqdm(raw_notes.iterrows(), total=len(raw_notes), desc="iclr2026 raw notes"):
        decision[raw_row.paper_id] = raw_row.decision
        try:
            submission = json.loads(raw_row.api_raw_submission)
            content = submission.get("content", {})
            area[raw_row.paper_id] = unwrap(content.get("primary_area"))
            final_venue[raw_row.paper_id] = unwrap(content.get("venue"))
        except TypeError, json.JSONDecodeError, AttributeError:
            pass
        try:
            note = json.loads(raw_row.api_raw_review)
        except TypeError, json.JSONDecodeError:
            continue
        if "id" not in note:
            continue
        questions[note["id"]] = unwrap(note.get("content", {}).get("questions")) or ""
        created[note["id"]] = note.get("cdate")
        modified[note["id"]] = note.get("mdate")
    snapshot_reviews["questions"] = snapshot_reviews.review_id.map(questions)
    text_fields = ["summary", "strengths", "weaknesses", "questions", "review"]
    snapshot_reviews[text_fields] = snapshot_reviews[text_fields].fillna("")
    text = snapshot_reviews[text_fields].apply(join_fields, axis=1)
    reviews = pd.DataFrame(
        dict(
            conf="ICLR",
            year=2026,
            paper_id=snapshot_reviews.paper_id,
            review_id=snapshot_reviews.review_id,
            reviewer=snapshot_reviews.reviewer,
            rating=snapshot_reviews.rating,
            confidence=snapshot_reviews.confidence,
            text=text,
            weaknesses=snapshot_reviews.weaknesses.map(squash),
            form="questions+strengths+summary+weaknesses",
            review_time=pd.to_datetime(snapshot_reviews.review_id.map(created), unit="ms", errors="coerce"),
            modified_time=pd.to_datetime(snapshot_reviews.review_id.map(modified), unit="ms", errors="coerce"),
        )
    )
    paper_rows = snapshot_reviews.drop_duplicates("paper_id")
    paper_final_venue = paper_rows.paper_id.map(final_venue).astype("str")
    left_conference = paper_final_venue.str.contains("Withdrawn|Desk Rejected", na=False)
    withdrawn_in_snapshot = paper_rows.venue.str.contains("Withdrawn")
    snapshot_decision = paper_rows.venue.where(withdrawn_in_snapshot, paper_rows.paper_id.map(decision))
    papers = pd.DataFrame(
        dict(
            conf="ICLR",
            year=2026,
            paper_id=paper_rows.paper_id,
            title=paper_rows.title,
            abstract=paper_rows.abstract,
            decision_raw=paper_final_venue.where(left_conference, snapshot_decision),
            primary_area=paper_rows.paper_id.map(area),
            final_venue=paper_final_venue,
        )
    )
    return reviews, papers


def is_older_version(reviews):
    repeated = reviews[reviews.reviewer.notna() & reviews.duplicated(["paper_id", "reviewer"], keep=False)]
    older = set()
    for _, versions in repeated.sort_values("review_time", kind="stable").groupby(["paper_id", "reviewer"]):
        opening_words = versions.text.str.slice(0, NEAR_COPY_CHARS).str.split().tolist()
        for i, review_id in enumerate(versions.review_id):
            later_versions = opening_words[i + 1 :]
            if any(SequenceMatcher(None, opening_words[i], later).ratio() >= NEAR_COPY_RATIO for later in later_versions):
                older.add(review_id)
    return reviews.review_id.isin(older)


def merge_ml(parts):
    reviews = pd.concat([r for r, _ in parts], ignore_index=True)
    reviews = reviews[reviews.text.str.len() > MIN_REVIEW_CHARS].drop_duplicates("review_id")
    reviews = reviews.drop_duplicates(["paper_id", "text"])
    older = is_older_version(reviews)
    print("older versions of a reviewer's own review dropped:", reviews[older].groupby(["conf", "year"]).size().to_dict())
    reviews = reviews[~older]
    papers = pd.concat([p for _, p in parts], ignore_index=True).drop_duplicates("paper_id")
    papers["decision"] = papers.decision_raw.fillna("").map(decision_class)
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
    summary = (
        reviews[~reviews.withdrawn]
        .groupby(["conf", "year"])
        .agg(
            papers=("paper_id", "nunique"),
            reviews=("review_id", "size"),
            median_chars=("text", lambda s: int(s.str.len().median())),
            has_conf=("confidence", lambda s: s.notna().mean().round(2)),
            has_time=("review_time", lambda s: s.notna().mean().round(2)),
        )
    )
    summary["rev_per_paper"] = (summary.reviews / summary.papers).round(2)
    print(summary.to_string())
    print(papers.groupby(["conf", "year"]).decision.value_counts().unstack(fill_value=0).to_string())


def inventory(reviews, papers, sources, n_dups, src):
    header = (
        f"data/other_reviews.parquet: {len(reviews)} reviews, {reviews.paper_id.nunique()} papers; sources: {', '.join(sources)}"
    )
    lines = [header, ""]
    by_year = reviews.groupby(["conf", "year"]).agg(
        papers=("paper_id", "nunique"),
        reviews=("review_id", "size"),
        median_chars=("n_chars", "median"),
        share_rated=("rating", lambda r: r.notna().mean()),
    )
    by_year.insert(2, "rev_per_paper", by_year.reviews / by_year.papers)
    totals = reviews.groupby("conf").agg(papers=("paper_id", "nunique"), reviews=("review_id", "size"))
    lines += ["Per source x year", by_year.round(2).to_string(), ""]
    lines += ["Totals by source", totals.to_string(), ""]
    lines.append(f"Exact-duplicate texts within a paper in the merged file: {n_dups} (dropped)")
    stats = json.loads((src / "stats.json").read_text())
    for source in sources:
        if source in stats:
            lines.append(f"  {source} fetch stats: {stats[source]}")
    if "plos" in stats:
        plos = stats["plos"]
        lines.append(
            f"PLOS: letters where reviewer splitting found 2+ reviewers: {plos['letters_split_2plus']}/{plos['letters']}"
            f" = {plos['letters_split_2plus'] / plos['letters']:.1%}"
        )
    licences = papers.groupby(["conf", "licence"], dropna=False).size()
    journal_papers = reviews.groupby(["conf", "journal"]).paper_id.nunique().sort_values(ascending=False)
    top_journals = journal_papers.groupby(level=0).head(TOP_JOURNALS_PER_SOURCE)
    lines += ["", "Licences (papers)", licences.to_string(), ""]
    lines += ["Journals", top_journals.to_string(), ""]
    lines.append(f"Random snippets (first {SNIPPET_CHARS} chars)")
    for conf, conf_reviews in reviews.groupby("conf"):
        for review in conf_reviews.sample(SNIPPETS_PER_SOURCE, random_state=0).itertuples():
            label = f"{conf} {review.year} {review.paper_id} {review.reviewer} rating={review.rating_raw}"
            lines.append(f"[{label}] {review.text[:SNIPPET_CHARS]}")
        lines.append("")
    return "\n".join(lines)


def other(src=settings.OTHER_REVIEWS_DIR, out=settings.DATA_DIR, report=REPORT):
    src, out = Path(src), Path(out)
    found_sources = [s for s in SOURCES if (src / f"{s}_reviews.parquet").exists()]
    reviews = pd.concat([pd.read_parquet(src / f"{s}_reviews.parquet") for s in found_sources], ignore_index=True)
    papers = pd.concat([pd.read_parquet(src / f"{s}_papers.parquet") for s in found_sources], ignore_index=True)
    for df in (reviews, papers):
        df["journal"] = df.journal.str.replace(r"^PLoS\b", "PLOS", regex=True).replace({"PLOS One": "PLOS ONE"})
    reviews["rating"] = pd.to_numeric(reviews.rating).round().astype("Int64")
    reviews["rating_raw"] = reviews.rating_raw.astype("string")
    reviews["year"] = reviews.year.astype(int)
    reviews = reviews[REVIEW_COLS]
    n_dups = int(reviews.duplicated(["paper_id", "text"]).sum())
    reviews = reviews.drop_duplicates(["paper_id", "text"])
    assert reviews.review_id.is_unique
    reviews_per_paper = reviews.groupby("paper_id").size()
    kept_counts = reviews_per_paper[reviews_per_paper >= MIN_REVIEWS_PER_PAPER]
    reviews = reviews[reviews.paper_id.isin(kept_counts.index)].reset_index(drop=True)
    papers = papers[papers.paper_id.isin(kept_counts.index)]
    papers["n_reviews"] = papers.paper_id.map(kept_counts)
    papers = papers[PAPER_COLS].reset_index(drop=True)
    reviews.to_parquet(out / "other_reviews.parquet")
    text = inventory(reviews, papers, found_sources, n_dups, src)
    print(text)
    Path(report).write_text(text + "\n")


if __name__ == "__main__":
    fire.Fire(dict(ml=ml, other=other))
