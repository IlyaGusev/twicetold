"""Fetch ICML 2026 accepted papers with their public reviews from OpenReview (API v2) for the policy comparison (RQ4).

Needs OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in the environment (anonymous API access is behind a challenge).
Writes data/icml2026_raw.jsonl (one submission per line, with replies) and data/icml2026_reviews.parquet with
paper_id, review_id, llm_policy (the submission's "LLM Policy" field), text (fields joined with tags), rating,
confidence, primary_area, review_time, modified_time.
"""

import json
import os
from pathlib import Path

import openreview
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
VENUE = "ICML.cc/2026/Conference"
FIELD_TAGS = [("summary", "SUMMARY"), ("strengths_and_weaknesses", "STRENGTHS AND WEAKNESSES"),
              ("key_questions_for_authors", "QUESTIONS"), ("limitations", "LIMITATIONS")]


def val(content, key):
    v = content.get(key)
    return v.get("value") if isinstance(v, dict) else v


def fetch():
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net",
                                             username=os.environ["OPENREVIEW_USERNAME"], password=os.environ["OPENREVIEW_PASSWORD"])
    notes = client.get_all_notes(content={"venueid": VENUE}, details="replies")
    with open(DATA / "icml2026_raw.jsonl", "w") as f:
        for n in notes:
            f.write(json.dumps(dict(id=n.id, content=n.content, replies=n.details.get("replies", []))) + "\n")
    print("submissions", len(notes))


def build():
    rows = []
    for line in open(DATA / "icml2026_raw.jsonl"):
        s = json.loads(line)
        policy = val(s["content"], "llm_policy") or next((val(s["content"], k) for k in s["content"] if "policy" in k.lower()), None)
        for r in s["replies"]:
            if not any(inv.endswith("/Official_Review") for inv in r.get("invitations", [])):
                continue
            c = r["content"]
            text = "\n\n".join(f"[{tag}]\n{val(c, k)}" for k, tag in FIELD_TAGS if val(c, k))
            rows.append(dict(conf="ICML", year=2026, paper_id=s["id"], review_id=r["id"], llm_policy=policy, text=text,
                             rating=val(c, "overall_recommendation"), confidence=val(c, "confidence"),
                             compliance=val(c, "compliance_with_llm_reviewing_policy"),
                             primary_area=val(s["content"], "primary_area"), review_time=r.get("cdate"), modified_time=r.get("mdate")))
    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "icml2026_reviews.parquet")
    print(df.groupby("llm_policy").paper_id.nunique())


if __name__ == "__main__":
    if not (DATA / "icml2026_raw.jsonl").exists():
        fetch()
    build()
