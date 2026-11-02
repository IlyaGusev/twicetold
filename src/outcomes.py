import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from extract import CATEGORIES
from metrics import assign, load_jsonl, panel_metrics
from metrics import overlap as hit
from settings import settings

PANEL_FILES = ["panels.jsonl", "panels_icml.jsonl", "panels_other.jsonl"]
WITHDRAWN_TAG = "_wd"
DROP_REASON = "a review of a withdrawn or desk-rejected paper, or a review not in reviews.parquet"
MIN_PLACEBO_REVIEWS = 2
PLACEBO_DROP_REASON = f"fewer than {MIN_PLACEBO_REVIEWS} reviews left after removing reviews of withdrawn or desk-rejected papers"
DROP_REASONS = {"panel": DROP_REASON, "sub3": DROP_REASON, "placebo": PLACEBO_DROP_REASON}
SOURCE_FILES = {"_icml": "icml_sample.parquet", "_other": "other_reviews.parquet"}
SOURCE_PANEL_FILES = {"_icml": ["panels_icml.jsonl"], "_other": ["panels_other.jsonl", "panels_other_placebo.jsonl"]}
STALE_REASON = "a cluster row that does not match the current panel files and source parquet"
SOURCE_EXTRA_COLUMNS = {"_icml": [], "_other": ["field", "journal"]}
MARKERS = [
    "commendable",
    "meticulous",
    "intricate",
    "delve",
    "showcas",
    "noteworthy",
    "underscore",
    "pivotal",
    "seamless",
    "invaluable",
    "notably",
    "realm",
    "comprehensive",
    "crucial",
    "robust",
    "leverag",
    "nuanced",
    "multifaceted",
    "paramount",
    "elucidat",
    "garner",
    "tapestry",
    "landscape",
    "holistic",
    "valuable insights",
]
MARKER_RX = re.compile(r"\b(" + "|".join(MARKERS) + ")", re.IGNORECASE)
EDIT_RX = re.compile(
    r"(after|post)[- ]?(the )?(rebuttal|discussion|response)|concerns? (have been|were|was|are) (addressed|resolved)"
    r"|thank(s| you)[^.]{0,40}(response|rebuttal|clarification)|(rais|updat|increas|lower|revis)(e|ed|ing) (my|the) (score|rating|review)"
    r"|\[?(update|revision|edit)d?[:\]]",
    re.IGNORECASE,
)
BATCH_ROWS = 20000
ICLR_FROM = 2020
FIRST_YEAR = {"ICLR": ICLR_FROM}
DEFAULT_FIRST_YEAR = 2019


def load_claims(path):
    fields = ("category", "target", "specific")
    claims = {}
    with open(path) as f:
        for row in map(json.loads, f):
            if "claims" in row:
                review_id = row["review_id"]
                claims[review_id] = [{field: claim.get(field) for field in fields} for claim in row["claims"]]
    return claims


def batches(path, columns, ids, desc):
    parquet = pq.ParquetFile(path)
    wanted = pa.array(list(ids), type=parquet.schema_arrow.field("review_id").type)
    n_batches = -(-parquet.metadata.num_rows // BATCH_ROWS)
    for batch in tqdm(parquet.iter_batches(batch_size=BATCH_ROWS, columns=columns), total=n_batches, desc=desc):
        yield batch.filter(pc.is_in(batch.column("review_id"), value_set=wanted))


def review_meta(ids, data=settings.DATA_DIR):
    parts = []
    cols = ["review_id", "paper_id", "conf", "year", "rating", "confidence", "text", "review_time", "modified_time"]
    for batch in batches(data / "reviews.parquet", cols, ids, "review text"):
        chunk = batch.to_pandas()
        chunk["chars"] = chunk.text.str.len()
        chunk["n_markers"] = chunk.text.map(lambda t: len(MARKER_RX.findall(t)))
        chunk["rebuttal_text"] = chunk.text.map(lambda t: bool(EDIT_RX.search(t)))
        parts.append(chunk.drop(columns="text"))
    reviews = pd.concat(parts, ignore_index=True)
    reviews["any_marker"] = reviews.n_markers > 0
    reviews["marker_rate"] = reviews.n_markers / (reviews.chars / 1000)
    ratings = pd.read_parquet(data / "review_cdates.parquet", columns=["review_id", "ext_rating_before", "ext_rating_after"])
    rating_changed = ratings.dropna().query("ext_rating_before != ext_rating_after")
    reviews["edited"] = reviews.review_id.isin(rating_changed.review_id)
    return reviews


def paper_meta(data=settings.DATA_DIR):
    papers = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "conf", "year", "decision", "n_reviews"])
    papers = papers.rename(columns={"n_reviews": "paper_n_reviews"})
    areas = pd.read_parquet(data / "paper_areas.parquet", columns=["paper_id", "area", "area_prob"])
    return papers.merge(areas, on="paper_id", how="left")


def in_main_sample(frame):
    return ~frame.withdrawn & frame.decision.ne("desk_reject")


def review_status(data):
    return pd.read_parquet(data / "reviews.parquet", columns=["review_id", "withdrawn", "decision"])


def excluded_review_ids(data):
    reviews = review_status(data)
    return set(reviews.review_id[~in_main_sample(reviews)])


def main_sample_review_ids(data):
    reviews = review_status(data)
    return set(reviews.review_id[in_main_sample(reviews)])


def review_rows(df, review_ids, paper):
    rows = list(df.itertuples(index=False))
    cluster_reviewers = Counter(cluster for _, cluster in {(row.review, row.cluster) for row in rows})
    target_reviewers = Counter(target for _, target in {(row.review, row.target) for row in rows})
    for j, review_id in enumerate(review_ids):
        claims = [row for row in rows if row.review == j]
        n = len(claims)
        categories = Counter(row.category for row in claims)
        shared = sum(cluster_reviewers[row.cluster] >= 2 for row in claims)
        yield dict(
            review_id=review_id,
            paper_id=paper,
            n_claims=n,
            share_shared=shared / n if n else np.nan,
            share_shared_lenient=sum(target_reviewers[row.target] >= 2 for row in claims) / n if n else np.nan,
            n_unique=n - shared,
            share_specific=sum(row.specific for row in claims) / n if n else np.nan,
            **{f"cat_{c}": categories[c] / n if n else np.nan for c in CATEGORIES},
        )


def count_true(flags):
    return int(flags.fillna(False).sum())


def claims_of_reviews(df, review_ids):
    if df.empty:
        return df
    position = {review_id: j for j, review_id in enumerate(review_ids)}
    kept = df[df.review_id.isin(review_ids)]
    return kept.assign(review=kept.review_id.map(position))


def ml_panels(model, tag, data, out):
    claims = load_claims(data / f"claims_{model}{tag}.jsonl")
    rmeta = review_meta(claims.keys(), data).set_index("review_id")
    pmeta = paper_meta(data).set_index("paper_id")
    corpus_reviews = set(review_status(data).review_id)
    excluded = set() if tag == WITHDRAWN_TAG else excluded_review_ids(data)

    rows_by_kind = {"panel": [], "placebo": [], "sub3": []}
    dropped = Counter()
    reviews = []
    for panel_id, panel in tqdm(load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id").items(), desc="panels"):
        kind = "placebo" if panel_id.startswith("placebo") else "sub3" if panel_id.startswith("sub3_") else "panel"
        paper = re.sub(r"^(noq|wd|sub3)_", "", panel_id)
        if kind == "panel" and paper not in pmeta.index:
            continue
        if kind == "placebo":
            review_ids = [r for r in panel["review_ids"] if r in corpus_reviews and r not in excluded]
            keep = len(review_ids) >= MIN_PLACEBO_REVIEWS
        else:
            review_ids = panel["review_ids"]
            keep = corpus_reviews.issuperset(review_ids) and excluded.isdisjoint(review_ids)
        if not keep:
            dropped[kind] += 1
            continue
        df = claims_of_reviews(assign(panel, claims), review_ids)
        row = panel_metrics(df, len(review_ids))
        panel_reviews = rmeta.reindex(review_ids)
        row.update(
            panel_id=panel_id,
            total_chars=float(panel_reviews.chars.sum()),
            rating_mean=panel_reviews.rating.mean(),
            rating_sd=panel_reviews.rating.std(ddof=0),
            confidence_mean=panel_reviews.confidence.mean(),
            n_marker_reviews=count_true(panel_reviews.any_marker),
            n_rebuttal_text=count_true(panel_reviews.rebuttal_text),
            n_edited=count_true(panel_reviews.edited),
        )
        if kind == "placebo":
            _, conf, year, _ = panel_id.split("_")
            row.update(
                conf=conf,
                year=int(year),
                placebo_kind="same_area3" if panel_id.startswith("placeboA_") else "shuffled4",
                n_excluded_reviews=len(panel["review_ids"]) - len(review_ids),
            )
        else:
            row.update(pmeta.loc[paper].to_dict(), paper_id=paper)
        rows_by_kind[kind].append(row)
        if kind == "panel" and not df.empty:
            reviews.extend(review_rows(df, review_ids, paper))
    tables = {kind: pd.DataFrame(rows) for kind, rows in rows_by_kind.items() if rows}
    for kind, table in tables.items():
        table.to_parquet(out / f"{kind}_outcomes_{model}{tag}.parquet")
        print(kind, len(table), "panels,", dropped[kind], "dropped:", DROP_REASONS[kind])
    if "panel" in tables:
        review_table = pd.DataFrame(reviews)
        review_table = review_table.merge(rmeta.drop(columns=["paper_id"]), left_on="review_id", right_index=True, how="left")
        review_table = review_table.merge(pmeta[["decision", "area"]], left_on="paper_id", right_index=True, how="left")
        review_table.to_parquet(out / f"review_outcomes_{model}{tag}.parquet")
        print("reviews", len(review_table))


def panel_review_ids(data, names):
    review_ids = {}
    for name in names:
        with open(data / name) as f:
            review_ids.update((panel["panel_id"], panel["review_ids"]) for panel in map(json.loads, f))
    return review_ids


def current_source_panels(data, tag, source_reviews):
    current = panel_review_ids(data, SOURCE_PANEL_FILES[tag])
    return {panel_id: review_ids for panel_id, review_ids in current.items() if source_reviews.issuperset(review_ids)}


def source_panels(model, tag, data, out):
    source = tag[1:]
    extra_columns = SOURCE_EXTRA_COLUMNS[tag]
    claims = load_claims(data / f"claims_{model}{tag}.jsonl")
    meta_columns = ["review_id", "conf", "year", "rating"] + extra_columns
    meta = pd.read_parquet(data / SOURCE_FILES[tag], columns=meta_columns).set_index("review_id")
    current = current_source_panels(data, tag, set(meta.index))
    n_stale = 0
    rows = []
    for panel_id, panel in tqdm(load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id").items(), desc=source):
        if current.get(panel_id) != panel["review_ids"]:
            n_stale += 1
            continue
        row = panel_metrics(assign(panel, claims), len(panel["review_ids"]), ks=(2,))
        first_review = meta.loc[panel["review_ids"][0]]
        row.update(
            panel_id=panel_id,
            placebo=panel_id.startswith("otherplacebo_"),
            conf=first_review.conf,
            year=int(first_review.year),
            rating_sd=meta.rating.reindex(panel["review_ids"]).std(ddof=0),
            **first_review[extra_columns].to_dict(),
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_parquet(out / f"{source}_panel_outcomes_{model}.parquet")
    print(len(rows), "panels,", n_stale, "dropped:", STALE_REASON)
    print(table.groupby(["placebo", "conf", "year"]).hit_rate_strict.mean().mul(100).round(2).unstack(0).to_string())


def panels(
    model=settings.DEFAULT_MODEL,
    tag="",
    data=str(settings.DATA_DIR),
    out=str(settings.ANALYSIS_DIR),
):
    Path(out).mkdir(parents=True, exist_ok=True)
    if tag in SOURCE_FILES:
        source_panels(model, tag, Path(data), Path(out))
    else:
        ml_panels(model, tag, Path(data), Path(out))


def panel_pairs(df, review_ids):
    groups = [set() for _ in review_ids]
    specific = [set() for _ in review_ids]
    for claim in df.itertuples(index=False):
        groups[claim.review].add(claim.cluster)
        if claim.specific:
            specific[claim.review].add(claim.cluster)
    for a, b in combinations(range(len(review_ids)), 2):
        if groups[a] and groups[b]:
            yield dict(a=review_ids[a], b=review_ids[b], hit=hit(groups[a], groups[b]), hit_spec=hit(specific[a], specific[b]))


def pairs(model=settings.DEFAULT_MODEL, data=str(settings.DATA_DIR), out=str(settings.ANALYSIS_DIR)):
    data = Path(data)
    Path(out).mkdir(parents=True, exist_ok=True)
    papers = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    main_reviews = main_sample_review_ids(data)
    n_dropped = n_stale = 0
    rows = []
    for tag in ["", *SOURCE_FILES]:
        clusters_path = data / f"clusters_{model}{tag}.jsonl"
        if not clusters_path.exists():
            continue
        claims = load_claims(data / f"claims_{model}{tag}.jsonl")
        if tag:
            meta = pd.read_parquet(data / SOURCE_FILES[tag], columns=["review_id", "conf", "year"]).set_index("review_id")
            current = current_source_panels(data, tag, set(meta.index))
        for panel_id, panel in tqdm(load_jsonl(clusters_path, "panel_id").items(), desc=f"pairs{tag}"):
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
            if df.empty:
                continue
            rows.extend(
                dict(panel_id=panel_id, conf=conf, year=int(year), **pair) for pair in panel_pairs(df, panel["review_ids"])
            )
    table = pd.DataFrame(rows)
    table.to_parquet(Path(out) / f"pairs_{model}.parquet")
    print(n_dropped, "panels dropped:", DROP_REASON)
    print(n_stale, "panels dropped:", STALE_REASON)
    print(table.groupby(["conf", "year"]).agg(pairs=("hit", "size"), hit=("hit", "mean")).to_string())


def review_chars(data, ids):
    parts = []
    for name in ["reviews.parquet", *SOURCE_FILES.values()]:
        for batch in batches(data / name, ["review_id", "text"], ids, name):
            lengths = pc.utf8_length(batch.column("text")).to_numpy()
            parts.append(pd.Series(lengths, index=batch.column("review_id").to_pandas()))
    chars = pd.concat(parts)
    return chars[~chars.index.duplicated()]


def in_window(df):
    return df[df.year >= df.conf.map(FIRST_YEAR).fillna(DEFAULT_FIRST_YEAR)]


def pair_design(model, data):
    design = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    design = in_window(design)
    chars = review_chars(data, set(design.a) | set(design.b))
    log_a = np.log(chars.reindex(design.a).to_numpy())
    log_b = np.log(chars.reindex(design.b).to_numpy())
    design["log_short"], design["log_long"] = np.minimum(log_a, log_b), np.maximum(log_a, log_b)
    sizes = {panel_id: len(review_ids) for panel_id, review_ids in panel_review_ids(data, PANEL_FILES).items()}
    design["size"] = design.panel_id.map(sizes)
    design["size"] = design["size"].clip(upper=design.groupby("conf")["size"].transform("min") + 2)
    return design


if __name__ == "__main__":
    fire.Fire({"panels": panels, "pairs": pairs})
