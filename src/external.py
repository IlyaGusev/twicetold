import re

import fire
import pandas as pd

import hub
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


def intel_text(row, columns):
    parts = []
    for tag, keys in INTEL_FIELDS:
        for key in keys:
            if key in columns and isinstance(row.get(key), str) and row[key].strip():
                parts.append(f"[{tag.upper()}] " + re.sub(r"\s+", " ", row[key]).strip())
    return " ".join(parts)


def intel(out):
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
    ours = hub.corpus_reviews(["review_id", "paper_id"])
    human = reviews[reviews.label == 0].merge(ours, left_on="orig_review_id", right_on="review_id", suffixes=("", "_r"))
    paper_map = human.drop_duplicates("intel_paper").set_index(["year", "intel_paper"]).paper_id
    reviews["paper_id"] = reviews.set_index(["year", "intel_paper"]).index.map(paper_map)
    reviews.to_parquet(out)
    print(len(reviews), "reviews,", reviews.paper_id.notna().sum(), "with a paper id")


if __name__ == "__main__":
    fire.Fire({"intel": intel})
