import json
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm

import extract
import hub
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
REWRITE_ML_PAPERS = 150
REWRITE_ML_EXTRA_PAPERS = 450
REWRITE_JOURNAL_PAPERS = 40
REWRITE_ML_YEARS = (2020, 2022)
REWRITE_LAST_JOURNAL_YEAR = 2022
REWRITE_MIN_REVIEWS = 3
REWRITTEN_PER_PANEL = 2
RewriteCondition = namedtuple("RewriteCondition", "writer task panel preserve text sources")
REWRITE_CONDITIONS = {
    "outline": RewriteCondition(settings.AI_REFERENCE_MODEL, "rewrites", "rw2_", "pres_", "rw_", None),
    "polish": RewriteCondition(settings.AI_REFERENCE_MODEL, "polishes", "rwp_", "ppres_", "pl_", None),
    "outline_second": RewriteCondition(settings.REWRITE_SECOND_WRITER, "rewrites", "rk2_", "kpres_", "rk_", ("ICLR",)),
    "polish_second": RewriteCondition(settings.REWRITE_SECOND_WRITER, "polishes", "rkp_", "kppres_", "pk_", ("ICLR",)),
}
PRESERVE_AUDIT_REVIEWS = 100
PAIR_PREFIX = "pair_"
PAPER_REVIEW_WRITERS = (settings.AI_REFERENCE_MODEL, settings.REWRITE_SECOND_WRITER)
POLISH_PER_SOURCE = 300
ML_SOURCES = ("ICLR", "NeurIPS")
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
    ml = hub.corpus_reviews(cols, (ds.field("year") >= 2018) & (ds.field("year") <= 2021))
    other = hub.journal_reviews(cols, ds.field("year") <= 2021)
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


def write_rows(todo, worker, out, threads, desc):
    with open(out, "a") as f, ThreadPoolExecutor(threads) as pool:
        bar = tqdm(pool.map(worker, todo), total=len(todo), desc=desc)
        for res in bar:
            if res:
                f.write(json.dumps(res) + "\n")
            bar.set_postfix(cost=f"${extract.cost():.2f}")
    print(f"done, ${extract.cost():.2f}")


def reference(model=settings.AI_REFERENCE_MODEL, threads=32, per_source=3000, out=None):
    effort = extract.reasoning_efforts(model).pass1
    out = Path(out) if out else settings.DATA_DIR / f"ai_reference_{model}.jsonl"
    done = done_ids(out)
    todo = [r for r in reference_sample(per_source).itertuples() if r.review_id not in done]
    print(f"{len(done)} done, {len(todo)} to write -> {out}")
    write_rows(todo, partial(rewrite, model=model, effort=effort), out, threads, "reference")


def polish_one(row, model, effort):
    reply = extract.call(
        render("ai_polish"), row.text[: settings.REVIEW_TEXT_MAX_CHARS], REVIEW_SCHEMA, "review", model=model, effort=effort
    )
    if "error" in reply:
        return None
    return {"review_id": row.review_id, "conf": row.conf, "year": int(row.year), "text": reply["review"]}


def polish(model=settings.AI_REFERENCE_MODEL, threads=32, per_source=POLISH_PER_SOURCE, out=None):
    effort = extract.reasoning_efforts(model).pass1
    out = Path(out) if out else settings.DATA_DIR / f"ai_polish_{model}.jsonl"
    done = done_ids(out)
    todo = [r for r in reference_sample(per_source).itertuples() if r.review_id not in done]
    print(f"{len(done)} done, {len(todo)} to polish -> {out}")
    write_rows(todo, partial(polish_one, model=model, effort=effort), out, threads, "polish")


def paper_kind(conf):
    return "ml" if conf in ML_SOURCES else "general"


def rewrite_sample(out=settings.REWRITE_TEST_DIR, seed=0, model=settings.DEFAULT_MODEL):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cols = ["panel_id", "conf", "year", "n_reviews"]
    ml = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet", columns=cols)
    ml = ml[(ml.conf == "ICLR") & ml.year.between(*REWRITE_ML_YEARS)].sort_values("panel_id")
    journals = pd.read_parquet(settings.ANALYSIS_DIR / f"other_panel_outcomes_{model}.parquet", columns=cols + ["placebo"])
    journals = journals[
        ~journals.placebo & (journals.year <= REWRITE_LAST_JOURNAL_YEAR) & (journals.n_reviews >= REWRITE_MIN_REVIEWS)
    ].sort_values("panel_id")
    first = ml.sample(REWRITE_ML_PAPERS, random_state=seed)
    chosen = pd.concat(
        [
            first,
            journals.groupby("conf").sample(REWRITE_JOURNAL_PAPERS, random_state=seed),
            ml.drop(first.index).sample(REWRITE_ML_EXTRA_PAPERS, random_state=seed + 1),
        ]
    )
    clusters = hub.clusters(model)
    rows = []
    for p in chosen.itertuples():
        review_ids = clusters[p.panel_id]["review_ids"]
        picked = [str(r) for r in rng.choice(review_ids, REWRITTEN_PER_PANEL, replace=False)]
        rows.append(dict(panel_id=p.panel_id, conf=p.conf, year=int(p.year), review_ids=review_ids, rewritten=picked))
    extract.write_jsonl(out / "papers.jsonl", rows)
    print(chosen.groupby("conf").size().to_string())


def source_texts(review_ids):
    cols = ["review_id", "conf", "year", "text"]
    ml = hub.sample("main", cols)
    other = hub.journal_reviews(cols)
    texts = pd.concat([ml[ml.review_id.isin(review_ids)], other[other.review_id.isin(review_ids)]])
    return texts.drop_duplicates("review_id").set_index("review_id")


def applies(condition, conf):
    return condition.sources is None or conf in condition.sources


def rewrite_panels(out=settings.REWRITE_TEST_DIR, threads=32):
    out = Path(out)
    papers = list(extract.read_jsonl(out / "papers.jsonl"))
    workers = {"rewrites": rewrite, "polishes": polish_one}
    for condition in REWRITE_CONDITIONS.values():
        wanted = [r for p in papers if applies(condition, p["conf"]) for r in p["rewritten"]]
        texts = source_texts(set(wanted)).loc[wanted].reset_index()
        effort = extract.reasoning_efforts(condition.writer).pass1
        path = out / f"{condition.task}_{condition.writer}.jsonl"
        done = done_ids(path)
        todo = [r for r in texts.itertuples() if r.review_id not in done]
        print(f"{len(done)} done, {len(todo)} to write -> {path}")
        worker = partial(workers[condition.task], model=condition.writer, effort=effort)
        write_rows(todo, worker, path, threads, f"{condition.task} {condition.writer}")


def rewrite_inputs(out=settings.REWRITE_TEST_DIR, model=settings.DEFAULT_MODEL, seed=0):
    out = Path(out)
    papers = list(extract.read_jsonl(out / "papers.jsonl"))
    versions = {
        name: {r["review_id"]: r["text"] for r in extract.read_jsonl(out / f"{c.task}_{c.writer}.jsonl")}
        for name, c in REWRITE_CONDITIONS.items()
    }
    kinds = {"ml": [], "general": []}
    texts = {"ml": [], "general": []}
    preserve = {name: [] for name in REWRITE_CONDITIONS}
    for p in papers:
        kind = paper_kind(p["conf"])
        kinds[kind].append(dict(panel_id=f"rw0_{p['panel_id']}", review_ids=p["review_ids"]))
        for name, c in REWRITE_CONDITIONS.items():
            if not applies(c, p["conf"]) or not all(r in versions[name] for r in p["rewritten"]):
                continue
            swapped = [c.text + r if r in p["rewritten"] else r for r in p["review_ids"]]
            kinds[kind].append(dict(panel_id=f"{c.panel}{p['panel_id']}", review_ids=swapped))
            for r in p["rewritten"]:
                texts[kind].append(dict(review_id=c.text + r, text=versions[name][r]))
                pair = dict(panel_id=f"{c.preserve}{r}", review_ids=[r, c.text + r])
                kinds[kind].append(pair)
                preserve[name].append(pair)
    for kind in kinds:
        extract.write_jsonl(out / f"panels_{kind}.jsonl", kinds[kind])
        pd.DataFrame(texts[kind]).to_parquet(out / f"reviews_{kind}.parquet")
    rng = np.random.default_rng(seed)
    audit = []
    for pairs in preserve.values():
        picked = rng.choice(len(pairs), min(PRESERVE_AUDIT_REVIEWS, len(pairs)), replace=False)
        audit += [pairs[i] for i in sorted(picked)]
    extract.write_jsonl(out / "panels_audit.jsonl", audit)
    originals = {r for p in papers for r in p["review_ids"]}
    path = out / f"claims_{model}.jsonl"
    kept = list(extract.read_jsonl(path)) if path.exists() else []
    have = {row["review_id"] for row in kept}
    claims = hub.claims(model)
    extract.write_jsonl(path, kept + [dict(review_id=r, claims=c) for r, c in claims.items() if r in originals - have])
    print(len(papers), "papers;", {k: len(v) for k, v in texts.items()}, "rewritten reviews")


def rewrite_pairs(out=settings.REWRITE_TEST_DIR):
    out = Path(out)
    papers = list(extract.read_jsonl(out / "papers.jsonl"))
    kinds = {"ml": [], "general": []}
    for p in papers:
        kind = paper_kind(p["conf"])
        kinds[kind].append(dict(panel_id=f"{PAIR_PREFIX}rw0_{p['panel_id']}", review_ids=p["rewritten"]))
        for c in REWRITE_CONDITIONS.values():
            if applies(c, p["conf"]):
                ids = [c.text + r for r in p["rewritten"]]
                kinds[kind].append(dict(panel_id=f"{PAIR_PREFIX}{c.panel}{p['panel_id']}", review_ids=ids))
    for kind, panels in kinds.items():
        extract.write_jsonl(out / f"panels_pairs_{kind}.jsonl", panels)
        print(kind, len(panels), "pair panels")


def paper_review(row, model, effort):
    prompt = render("ai_review", venue=VENUE["ICLR"], words=row.words)
    reply = extract.call(prompt, row.text, REVIEW_SCHEMA, "review", model=model, effort=effort)
    if "error" in reply:
        return None
    return {"review_id": row.paper_id, "conf": "ICLR", "year": int(row.year), "text": reply["review"]}


def paper_reviews(out=settings.PAPER_REVIEW_DIR, threads=16):
    out = Path(out)
    papers = pd.read_json(out / "papers.jsonl", lines=True)
    for writer in PAPER_REVIEW_WRITERS:
        path = out / f"reviews_{writer}.jsonl"
        done = done_ids(path)
        todo = [r for r in papers.itertuples() if r.paper_id not in done]
        print(f"{len(done)} done, {len(todo)} to write -> {path}")
        worker = partial(paper_review, model=writer, effort=extract.reasoning_efforts(writer).pass1)
        write_rows(todo, worker, path, threads, f"paper reviews {writer}")


if __name__ == "__main__":
    fire.Fire(
        {
            "reference": reference,
            "polish": polish,
            "rewrite_sample": rewrite_sample,
            "rewrite_panels": rewrite_panels,
            "rewrite_inputs": rewrite_inputs,
            "rewrite_pairs": rewrite_pairs,
            "paper_reviews": paper_reviews,
        }
    )
