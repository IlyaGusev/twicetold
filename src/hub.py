import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from settings import settings

NAMES = {
    "conf": "source",
    "score": "meld_score",
    "n_tokens": "meld_tokens",
    "family": "meld_family",
    "task": "meld_task",
    "flag": "meld_flag",
    "written": "meld_written",
}

JOURNAL_PANEL_KINDS = ["journal", "journal_placebo"]
STREAM = dict(batch_size=20000, batch_readahead=1, fragment_readahead=1)


def check_model(model):
    if model != settings.DEFAULT_MODEL:
        raise SystemExit(f"{settings.HF_DATASET_REPO} holds {settings.DEFAULT_MODEL} outputs only, not {model}")


def arrow(config, columns, condition=None):
    return ds.dataset(settings.FINAL_DIR / config, format="parquet").to_table(columns=columns, filter=condition)


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
    base = ds.field("source").isin(settings.ML_SOURCES) & in_corpus()
    return frame("reviews", columns, base if condition is None else base & condition, "corpus_row")


def journal_reviews(columns, condition=None):
    base = ds.field("venue_type") == "journal"
    return frame("reviews", columns, base if condition is None else base & condition, "corpus_row")


def sample(name, columns):
    return frame("reviews", columns, ds.field("sample") == name, "sample_row")


def variant(name, columns):
    return frame("variants", columns, ds.field("variant") == name, "sample_row")


def papers(columns):
    return frame("papers", columns, ds.field("source").isin(settings.ML_SOURCES), "corpus_row")


def paper_areas():
    return frame("papers", ["paper_id", "area_pred"], ds.field("area_pred").is_valid())


def meld_flags(columns=("conf", "year", "score", "flag", "written")):
    return frame("reviews", ["review_id", *columns], ds.field("meld_flag").is_valid()).set_index("review_id")


def editlens_flags():
    return (
        frame("reviews", ["review_id", "editlens_flag"], ds.field("editlens_flag").is_valid())
        .set_index("review_id")
        .editlens_flag
    )


def texts(review_ids):
    ids = list(review_ids)
    tables = [arrow(config, ["review_id", "text"], ds.field("review_id").isin(ids)) for config in ["reviews", "variants"]]
    return {
        review_id: text
        for table in tables
        for review_id, text in zip(table.column("review_id").to_pylist(), table.column("text").to_pylist())
    }


def text_chars(review_ids):
    wanted = ds.field("review_id").isin(pa.array(list(review_ids), pa.string()))
    parts = []
    for config in ["reviews", "variants"]:
        dataset = ds.dataset(settings.FINAL_DIR / config, format="parquet")
        for batch in dataset.to_batches(columns=["review_id", "text"], filter=wanted, **STREAM):
            parts.append(pd.Series(pc.utf8_length(batch.column("text")).to_numpy(), index=batch.column("review_id").to_pandas()))
    return pd.concat(parts)


def sampled_reviews(journal, without_claims=False):
    pending = ~ds.field("claims").is_valid() if without_claims else None
    in_sample = ds.field("sample").is_valid() & ((ds.field("venue_type") == "journal") == journal)
    tables = [arrow("reviews", ["review_id", "text"], in_sample if pending is None else in_sample & pending)]
    if not journal:
        tables.append(arrow("variants", ["review_id", "text"], pending))
    return [row for table in tables for row in table.to_pylist()]


def reviews_with_claims():
    tables = [arrow(config, ["review_id"], ds.field("claims").is_valid()) for config in ["reviews", "variants"]]
    return {review_id for table in tables for review_id in table.column("review_id").to_pylist()}


def projected_claims(lists, fields):
    values = lists.flatten()
    rows = list(zip(*(values.field(name).to_pylist() for name in fields)))
    ends = np.cumsum(lists.value_lengths().to_numpy())
    return [[dict(zip(fields, row)) for row in rows[end - size : end]] for end, size in zip(ends, np.diff(ends, prepend=0))]


def claims(model, review_ids=None, fields=None):
    check_model(model)
    condition = ds.field("claims").is_valid()
    if review_ids is not None:
        condition = condition & ds.field("review_id").isin(pa.array(list(review_ids), pa.string()))
    out = {}
    for config in ["reviews", "variants"]:
        dataset = ds.dataset(settings.FINAL_DIR / config, format="parquet")
        for batch in dataset.to_batches(columns=["review_id", "claims"], filter=condition, **STREAM):
            lists = batch.column("claims")
            out.update(
                zip(
                    batch.column("review_id").to_pylist(),
                    lists.to_pylist() if fields is None else projected_claims(lists, fields),
                )
            )
    return out


def texts_with_claims(model, fields):
    check_model(model)
    for config in ["reviews", "variants"]:
        dataset = ds.dataset(settings.FINAL_DIR / config, format="parquet")
        columns = ["review_id", "text", "claims"]
        for batch in dataset.to_batches(columns=columns, filter=ds.field("claims").is_valid(), **STREAM):
            claims_of_batch = projected_claims(batch.column("claims"), fields)
            yield from zip(batch.column("review_id").to_pylist(), batch.column("text").to_pylist(), claims_of_batch)


def claim_frames(model, fields):
    check_model(model)
    for config in ["reviews", "variants"]:
        dataset = ds.dataset(settings.FINAL_DIR / config, format="parquet")
        for batch in dataset.to_batches(columns=["review_id", "claims"], filter=ds.field("claims").is_valid(), **STREAM):
            lists = batch.column("claims")
            values = lists.flatten()
            sizes = lists.value_lengths().to_numpy()
            frame = pd.DataFrame({name: values.field(name).to_pandas() for name in fields})
            frame.insert(0, "review_id", np.repeat(batch.column("review_id").to_pandas().to_numpy(), sizes))
            frame.insert(1, "k", np.concatenate([np.arange(size) for size in sizes]) if len(sizes) else [])
            yield frame


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
    condition = ds.field("clusters").is_valid()
    if kinds is not None:
        condition = condition & ds.field("kind").isin(kinds)
    if panel_ids is not None:
        condition = condition & ds.field("panel_id").isin(list(panel_ids))
    rows = arrow("panels", ["panel_id", "review_ids", "targets", "clusters"], condition).to_pylist()
    return {row["panel_id"]: row for row in rows}


def panel_claims(model, panels, fields=None):
    return claims(model, {review_id for panel in panels.values() for review_id in panel["review_ids"]}, fields)


def panel_definitions(kinds):
    rows = arrow("panels", ["panel_id", "review_ids"], ds.field("kind").isin(kinds)).to_pylist()
    return {row["panel_id"]: row["review_ids"] for row in rows}


def panel_rows(journal):
    of_kind = ds.field("kind").isin(JOURNAL_PANEL_KINDS)
    return arrow("panels", ["panel_id", "review_ids"], of_kind if journal else ~of_kind).to_pylist()


def grouped_panels():
    rows = arrow("panels", ["panel_id", "review_ids"], ds.field("clusters").is_valid()).to_pylist()
    return {(row["panel_id"], tuple(row["review_ids"])) for row in rows}
