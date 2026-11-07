import sys
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.dataset as ds
from huggingface_hub import snapshot_download

from settings import settings

ML_SOURCES = ["ICLR", "NeurIPS"]
NAMES = {
    "conf": "source",
    "score": "meld_score",
    "n_tokens": "meld_tokens",
    "family": "meld_family",
    "task": "meld_task",
    "flag": "meld_flag",
    "written": "meld_written",
}


@cache
def root():
    if settings.HF_DATASET_LOCAL:
        return settings.RELEASE_DIR
    return Path(snapshot_download(settings.HF_DATASET_REPO, repo_type="dataset", revision=settings.HF_DATASET_REVISION))


def check_model(model):
    if model != settings.DEFAULT_MODEL:
        raise SystemExit(f"{settings.HF_DATASET_REPO} holds {settings.DEFAULT_MODEL} outputs only, not {model}")


def arrow(config, columns, condition=None):
    return ds.dataset(root() / config, format="parquet").to_table(columns=columns, filter=condition)


def frame(config, columns, condition=None, order=None):
    names = [NAMES.get(c, c) for c in columns]
    table = arrow(config, list(dict.fromkeys(names + ([order] if order else []))), condition)
    df = table.to_pandas(ignore_metadata=True)
    if order:
        df = df.sort_values(order, kind="stable").reset_index(drop=True)
    return df[names].set_axis(columns, axis=1)


def in_corpus():
    return ds.field("in_corpus")


def corpus_reviews(columns, condition=None):
    base = ds.field("source").isin(ML_SOURCES) & in_corpus()
    return frame("reviews", columns, base if condition is None else base & condition, "corpus_row")


def journal_reviews(columns, condition=None):
    base = ds.field("venue_type") == "journal"
    return frame("reviews", columns, base if condition is None else base & condition, "corpus_row")


def sample(name, columns):
    return frame("reviews", columns, ds.field("sample") == name, "sample_row")


def variant(name, columns):
    return frame("variants", columns, ds.field("variant") == name, "sample_row")


def papers(columns):
    return frame("papers", columns, ds.field("source").isin(ML_SOURCES), "corpus_row")


def paper_areas():
    return frame("papers", ["paper_id", "area_pred"], ds.field("area_pred").is_valid())


def meld_flags(columns):
    return frame("reviews", columns, ds.field("meld_flag").is_valid())


def texts(review_ids):
    ids = list(review_ids)
    tables = [arrow(config, ["review_id", "text"], ds.field("review_id").isin(ids)) for config in ["reviews", "variants"]]
    return {
        review_id: text
        for table in tables
        for review_id, text in zip(table.column("review_id").to_pylist(), table.column("text").to_pylist())
    }


def claims(model, review_ids=None):
    check_model(model)
    condition = ds.field("claims").is_valid()
    if review_ids is not None:
        condition = condition & ds.field("review_id").isin(list(review_ids))
    out = {}
    for config in ["reviews", "variants"]:
        table = arrow(config, ["review_id", "claims"], condition)
        out.update(zip(table.column("review_id").to_pylist(), table.column("claims").to_pylist()))
    return out


def extraction_counts(model):
    check_model(model)
    counts = dict(pass1_error_rows=0, pass1_empty=0, pass2_error_rows=0, pass2_rows=0)
    for config in ["reviews", "variants"]:
        table = arrow(config, ["claims", "claims_error"]).to_pandas()
        counts["pass1_error_rows"] += int(table.claims_error.notna().sum())
        done = table.claims.dropna()
        counts["pass1_empty"] += int(done.map(len).eq(0).sum())
    errors = arrow("panels", ["error"]).column("error")
    counts["pass2_error_rows"] = len(errors) - errors.null_count
    counts["pass2_rows"] = len(errors)
    return counts


def clusters(model, kinds=None, panel_ids=None):
    check_model(model)
    condition = ds.field("error").is_null()
    if kinds is not None:
        condition = condition & ds.field("kind").isin(kinds)
    if panel_ids is not None:
        condition = condition & ds.field("panel_id").isin(list(panel_ids))
    rows = arrow("panels", ["panel_id", "review_ids", "targets", "clusters"], condition).to_pylist()
    return {row["panel_id"]: row for row in rows}


def panel_claims(model, panels):
    return claims(model, {review_id for panel in panels.values() for review_id in panel["review_ids"]})


def panel_definitions(kinds):
    rows = arrow("panels", ["panel_id", "review_ids"], ds.field("kind").isin(kinds) & ds.field("current")).to_pylist()
    return {row["panel_id"]: row["review_ids"] for row in rows}
