import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import pandas as pd

from settings import settings

INTEL_FIELDS = [
    ("summary", ["summary", "summary_of_the_paper"]),
    ("main", ["review", "main_review"]),
    ("strengths", ["strengths", "strength_and_weaknesses"]),
    ("weaknesses", ["weaknesses"]),
    ("questions", ["questions"]),
    ("other", ["clarity,_quality,_novelty_and_reproducibility", "summary_of_the_review"]),
]
MIN_TEXT_CHARS = 200
CDATE_YEARS = (2023, 2024, 2025, 2026)
RATING_CHANGE_YEARS = (2024, 2025)


def intel_text(row, columns):
    parts = []
    for tag, keys in INTEL_FIELDS:
        for key in keys:
            if key in columns and isinstance(row.get(key), str) and row[key].strip():
                parts.append(f"[{tag.upper()}] " + re.sub(r"\s+", " ", row[key]).strip())
    return " ".join(parts)


def intel(out, data=str(settings.DATA_DIR)):
    rows = []
    for path in sorted(settings.INTEL_DIR.glob("test/*/ICLR*.csv")):
        llm = path.parent.name
        year = int(re.search(r"ICLR(\d{4})", path.name).group(1))
        d = pd.read_csv(path)
        d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
        d["text"] = d.apply(intel_text, axis=1, columns=set(d.columns))
        d["llm"] = llm
        d["year"] = year
        d["review_id"] = d.apply(lambda r: f"{'LLM_' + llm + '_' if r.label == 1 else ''}{r['review-id']}", axis=1)
        columns = ["review_id", "review-id", "paperid", "label", "llm", "year", "text"]
        rows.append(d[columns].rename(columns={"review-id": "orig_review_id", "paperid": "intel_paper"}))
    reviews = pd.concat(rows, ignore_index=True)
    reviews = reviews[reviews.text.str.len() > MIN_TEXT_CHARS]
    ours = pd.read_parquet(Path(data) / "reviews.parquet", columns=["review_id", "paper_id"])
    human = reviews[reviews.label == 0].merge(ours, left_on="orig_review_id", right_on="review_id", suffixes=("", "_r"))
    paper_map = human.drop_duplicates("intel_paper").set_index(["year", "intel_paper"]).paper_id
    reviews["paper_id"] = reviews.set_index(["year", "intel_paper"]).index.map(paper_map)
    reviews.to_parquet(out)
    print(len(reviews), "reviews,", reviews.paper_id.notna().sum(), "with a paper id")


def review_cdates(out, data=str(settings.DATA_DIR)):
    ours = pd.read_parquet(Path(data) / "reviews.parquet", columns=["review_id", "conf", "year", "paper_id", "reviewer"])
    dates = []
    for year in CDATE_YEARS:
        columns = ["forum", "primary_area", "reviewer_suffix", "review_cdate"]
        columns += ["ext_rating_before", "ext_rating_after"] if year in RATING_CHANGE_YEARS else []
        d = pd.read_parquet(settings.ICLR_AREAS_DIR / f"mlouis_{year}.parquet", columns=columns)
        d["reviewer"] = "Reviewer_" + d.reviewer_suffix.astype(str)
        dates.append(d.assign(year=year))
    dates = pd.concat(dates)
    matched = ours[ours.conf == "ICLR"].merge(
        dates, left_on=["paper_id", "reviewer", "year"], right_on=["forum", "reviewer", "year"], how="inner"
    )
    matched["review_cdate"] = pd.to_datetime(matched.review_cdate, unit="ms", errors="coerce")
    matched[["review_id", "review_cdate", "ext_rating_before", "ext_rating_after"]].to_parquet(out)
    print("matched by year:", matched.groupby("year").size().to_dict())


if __name__ == "__main__":
    fire.Fire({"intel": intel, "review_cdates": review_cdates})
