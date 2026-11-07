import re
import sys
from collections import Counter
from functools import cache
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import pandas as pd
from tqdm import tqdm

import hub
from metrics import assign, panel_metrics, review_groups
from metrics import overlap as hit
from settings import settings

WITHDRAWN_TAG = "_wd"
REVIEW_PREFIXES = {"_noq": "noq_", "_nocq": "nocq_"}
PANEL_PREFIXES = {
    "_icml": ("icml_",),
    "_other": ("other_", "otherplacebo_"),
    **{tag: (prefix,) for tag, prefix in REVIEW_PREFIXES.items()},
    WITHDRAWN_TAG: ("wd_",),
}
DROP_REASON = "a review of a withdrawn or desk-rejected paper, or a review not in reviews.parquet"
MIN_PLACEBO_REVIEWS = 2
PLACEBO_DROP_REASON = f"fewer than {MIN_PLACEBO_REVIEWS} reviews left after removing reviews of withdrawn or desk-rejected papers"
DROP_REASONS = {"panel": DROP_REASON, "sub3": DROP_REASON, "placebo": PLACEBO_DROP_REASON}
SOURCE_TAGS = ["_icml", "_other"]
SOURCE_PANEL_KINDS = {"_icml": ["icml"], "_other": ["journal", "journal_placebo"]}
STALE_REASON = "a cluster row that does not match the current panel files and source parquet"
ICLR_FROM = 2020
ICLR_LAST = 2027
FIRST_YEAR = {"ICLR": ICLR_FROM}
DEFAULT_FIRST_YEAR = 2019


def load_claims(model):
    return {review_id: [{"specific": claim["specific"]} for claim in claims] for review_id, claims in hub.claims(model).items()}


def tag_of(panel_id):
    return next((tag for tag, prefixes in PANEL_PREFIXES.items() if panel_id.startswith(prefixes)), "")


def load_outputs(model):
    clusters_by_tag = {}
    for panel_id, panel in hub.clusters(model).items():
        clusters_by_tag.setdefault(tag_of(panel_id), {})[panel_id] = panel
    return load_claims(model), clusters_by_tag


def in_main_sample(frame):
    return ~frame.withdrawn & frame.decision.ne("desk_reject")


@cache
def review_status():
    return hub.corpus_reviews(["review_id", "withdrawn", "decision"])


def excluded_review_ids():
    reviews = review_status()
    return set(reviews.review_id[~in_main_sample(reviews)])


def main_sample_review_ids():
    reviews = review_status()
    return set(reviews.review_id[in_main_sample(reviews)])


def ml_papers():
    return hub.papers(["paper_id", "conf", "year"]).set_index("paper_id")


def claims_of_reviews(df, review_ids):
    if df.empty:
        return df
    position = {review_id: j for j, review_id in enumerate(review_ids)}
    kept = df[df.review_id.isin(review_ids)]
    return kept.assign(review=kept.review_id.map(position))


def ml_panels(model, tag, clusters, claims, out):
    papers = ml_papers()
    corpus_reviews = set(review_status().review_id)
    excluded = set() if tag == WITHDRAWN_TAG else excluded_review_ids()

    rows_by_kind = {"panel": [], "placebo": [], "sub3": []}
    dropped = Counter()
    for panel_id, panel in tqdm(clusters.items(), desc=f"panels{tag}"):
        kind = "placebo" if panel_id.startswith("placebo") else "sub3" if panel_id.startswith("sub3_") else "panel"
        paper = re.sub(r"^(noq|nocq|wd|sub3)_", "", panel_id)
        if kind == "panel" and paper not in papers.index:
            continue
        if kind == "placebo":
            review_ids = [r for r in panel["review_ids"] if r in corpus_reviews and r not in excluded]
            keep = len(review_ids) >= MIN_PLACEBO_REVIEWS
        else:
            review_ids = panel["review_ids"]
            source_ids = [r.removeprefix(REVIEW_PREFIXES.get(tag, "")) for r in review_ids]
            keep = corpus_reviews.issuperset(source_ids) and excluded.isdisjoint(source_ids)
        if not keep:
            dropped[kind] += 1
            continue
        df = claims_of_reviews(assign(panel, claims), review_ids)
        row = dict(panel_metrics(df, len(review_ids)), panel_id=panel_id)
        if kind == "placebo":
            _, conf, year, _ = panel_id.split("_")
            row.update(conf=conf, year=int(year), placebo_kind="same_area3" if panel_id.startswith("placeboA_") else "shuffled4")
        else:
            row.update(papers.loc[paper].to_dict(), paper_id=paper)
        rows_by_kind[kind].append(row)
    for kind, rows in rows_by_kind.items():
        if rows:
            pd.DataFrame(rows).to_parquet(out / f"{kind}_outcomes_{model}{tag}.parquet")
            print(kind, len(rows), "panels,", dropped[kind], "dropped:", DROP_REASONS[kind])


def source_meta(tag):
    columns = ["review_id", "conf", "year"]
    meta = hub.sample("icml", columns) if tag == "_icml" else hub.journal_reviews(columns)
    return meta.set_index("review_id")


def current_source_panels(tag, source_reviews):
    panels = hub.panel_definitions(SOURCE_PANEL_KINDS[tag])
    return {panel_id: review_ids for panel_id, review_ids in panels.items() if source_reviews.issuperset(review_ids)}


def source_panels(model, tag, clusters, claims, out):
    source = tag[1:]
    meta = source_meta(tag)
    current = current_source_panels(tag, set(meta.index))
    n_stale = 0
    rows = []
    for panel_id, panel in tqdm(clusters.items(), desc=source):
        if current.get(panel_id) != panel["review_ids"]:
            n_stale += 1
            continue
        first_review = meta.loc[panel["review_ids"][0]]
        rows.append(
            dict(
                panel_metrics(assign(panel, claims), len(panel["review_ids"])),
                panel_id=panel_id,
                placebo=panel_id.startswith("otherplacebo_"),
                conf=first_review.conf,
                year=int(first_review.year),
            )
        )
    table = pd.DataFrame(rows)
    table.to_parquet(out / f"{source}_panel_outcomes_{model}.parquet")
    print(len(rows), "panels,", n_stale, "dropped:", STALE_REASON)
    print(table.groupby(["placebo", "conf", "year"]).hit_rate_strict.mean().mul(100).round(2).unstack(0).to_string())


def panels(model=settings.DEFAULT_MODEL, out=str(settings.ANALYSIS_DIR)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    claims, clusters_by_tag = load_outputs(model)
    for tag, clusters in clusters_by_tag.items():
        build = source_panels if tag in SOURCE_TAGS else ml_panels
        build(model, tag, clusters, claims, out)


def review_sets(df, n_reviews):
    return review_groups([], [], n_reviews) if df.empty else review_groups(df.review, df.cluster, n_reviews)


def panel_pairs(df, review_ids):
    groups = [set() for _ in review_ids]
    specific = [set() for _ in review_ids]
    for claim in df.itertuples(index=False):
        groups[claim.review].add(claim.cluster)
        if claim.specific:
            specific[claim.review].add(claim.cluster)
    for a, b in combinations(range(len(review_ids)), 2):
        yield dict(
            a=review_ids[a],
            b=review_ids[b],
            hit=hit(groups[a], groups[b]),
            hit_spec=hit(specific[a], specific[b]),
            shared=len(groups[a] & groups[b]),
            smaller=min(len(groups[a]), len(groups[b])),
            larger=max(len(groups[a]), len(groups[b])),
            union=len(groups[a] | groups[b]),
        )


def pairs(model=settings.DEFAULT_MODEL, out=str(settings.ANALYSIS_DIR)):
    Path(out).mkdir(parents=True, exist_ok=True)
    papers = ml_papers()
    main_reviews = main_sample_review_ids()
    n_dropped = n_stale = 0
    rows, reviews = [], []
    claims, clusters_by_tag = load_outputs(model)
    for tag in ["", *SOURCE_TAGS]:
        if tag:
            meta = source_meta(tag)
            current = current_source_panels(tag, set(meta.index))
        for panel_id, panel in tqdm(clusters_by_tag[tag].items(), desc=f"pairs{tag}"):
            if tag and panel_id.startswith(("icml_", "other_")):
                if current.get(panel_id) != panel["review_ids"]:
                    n_stale += 1
                    continue
                conf, year = meta.loc[panel["review_ids"][0], ["conf", "year"]]
            elif not tag and panel_id in papers.index:
                if not main_reviews.issuperset(panel["review_ids"]):
                    n_dropped += 1
                    continue
                conf, year = papers.loc[panel_id, ["conf", "year"]]
            else:
                continue
            df = assign(panel, claims)
            rows.extend(
                dict(panel_id=panel_id, conf=conf, year=int(year), **pair) for pair in panel_pairs(df, panel["review_ids"])
            )
            reviews.extend(
                dict(panel_id=panel_id, conf=conf, year=int(year), review_id=rid, groups=sorted(groups))
                for rid, groups in zip(panel["review_ids"], review_sets(df, len(panel["review_ids"])))
            )
    table = pd.DataFrame(rows)
    table.to_parquet(Path(out) / f"pairs_{model}.parquet")
    pd.DataFrame(reviews).to_parquet(Path(out) / f"review_groups_{model}.parquet")
    print(n_dropped, "panels dropped:", DROP_REASON)
    print(n_stale, "panels dropped:", STALE_REASON)
    print(table.groupby(["conf", "year"]).agg(pairs=("hit", "size"), hit=("hit", "mean")).to_string())


def in_window(df):
    return df[df.year >= df.conf.map(FIRST_YEAR).fillna(DEFAULT_FIRST_YEAR)]


if __name__ == "__main__":
    fire.Fire({"panels": panels, "pairs": pairs})
