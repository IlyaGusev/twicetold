import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import pandas as pd
from tqdm import tqdm

import extract
from prompts import render
from settings import settings

SECTION_TAGS = [
    "SUMMARY",
    "MAIN",
    "STRENGTHS",
    "WEAKNESSES",
    "QUESTIONS",
    "LIMITATIONS",
    "OTHER",
    "STRENGTHS AND WEAKNESSES",
    "CLAIMS AND EVIDENCE",
    "METHODS AND EVALUATION",
    "THEORY",
    "EXPERIMENTS",
    "SUPPLEMENT",
    "RELATED WORK",
    "MISSING REFERENCES",
    "REPORT",
    "REQUESTED CHANGES",
]
TAGS = r"\[(?:" + "|".join(SECTION_TAGS) + r")\]"

OUTLINE_SCHEMA = extract.strict_object(points={"type": "array", "items": {"type": "string"}})
REVIEW_SCHEMA = extract.strict_object(review={"type": "string"})
VENUE = {
    "ICLR": "a machine-learning conference (ICLR)",
    "NeurIPS": "a machine-learning conference (NeurIPS)",
    "SciPost": "a physics journal (SciPost)",
    "F1000": "an open-research biomedical journal (F1000Research)",
    "BMC-Med": "a medical journal (BMC Medicine)",
    "PLOS": "a multidisciplinary journal (PLOS ONE)",
    "Copernicus": "a geoscience journal (Copernicus/EGU)",
}


def done_ids(path):
    return {r["review_id"] for r in extract.read_jsonl(path)} if path.exists() else set()


def reference_sample(per_source):
    cols = ["conf", "year", "review_id", "text"]
    ml = pd.read_parquet(
        settings.DATA_DIR / "reviews.parquet", columns=cols, filters=[("year", ">=", 2018), ("year", "<=", 2021)]
    )
    other = pd.read_parquet(settings.DATA_DIR / "other_reviews.parquet", columns=cols, filters=[("year", "<=", 2021)])
    return pd.concat([ml, other]).sample(frac=1, random_state=0).groupby("conf").head(per_source)


def rewrite(row, model, effort):
    outline_prompt = render("ai_outline")
    outline_reply = extract.call(
        outline_prompt, row.text[: settings.REVIEW_TEXT_MAX_CHARS], OUTLINE_SCHEMA, "outline", model=model, effort=effort
    )
    if "error" in outline_reply:
        return None
    outline = "\n".join(f"- {point}" for point in outline_reply["points"])
    words = round(len(row.text.split()), -1)
    write_prompt = render("ai_write", venue=VENUE[row.conf], words=words)
    review_reply = extract.call(write_prompt, outline, REVIEW_SCHEMA, "review", model=model, effort=effort)
    if "error" in review_reply:
        return None
    return {
        "review_id": row.review_id,
        "conf": row.conf,
        "year": int(row.year),
        "outline": outline,
        "text": review_reply["review"],
    }


def reference(model=settings.AI_REFERENCE_MODEL, threads=32, per_source=3000, out=None):
    effort = extract.reasoning_efforts(model).pass1
    out = Path(out) if out else settings.DATA_DIR / f"ai_reference_{model}.jsonl"
    done = done_ids(out)
    todo = [r for r in reference_sample(per_source).itertuples() if r.review_id not in done]
    print(f"{len(done)} done, {len(todo)} to write -> {out}")
    with open(out, "a") as f, ThreadPoolExecutor(threads) as pool:
        bar = tqdm(pool.map(partial(rewrite, model=model, effort=effort), todo), total=len(todo), desc="reference")
        for res in bar:
            if res:
                f.write(json.dumps(res) + "\n")
            bar.set_postfix(cost=f"${extract.cost():.2f}")
    print(f"done, ${extract.cost():.2f}")


if __name__ == "__main__":
    fire.Fire({"reference": reference})
