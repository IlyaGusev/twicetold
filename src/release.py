import json
from functools import partial
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

from extract import merge_rows, panel_key, review_key
from jsonl import read_jsonl
from prompts import render
from settings import settings

OPENREVIEW_LICENCE = "OpenReview public review"
REVIEWER_LABEL = r"^(?:Reviewer|Referee)"
SAMPLES = {
    "main": "sample_reviews.parquet",
    "icml": "icml_sample.parquet",
    "journal": "other_pass1.parquet",
}
VARIANTS = {"noq": "noq_reviews.parquet", "nocq": "nocq_reviews.parquet"}
REVIEW_COLUMNS = [
    "source",
    "venue_type",
    "journal",
    "field",
    "year",
    "paper_id",
    "review_id",
    "reviewer",
    "rating",
    "rating_raw",
    "confidence",
    "text",
    "form",
    "review_time",
    "withdrawn",
    "decision",
    "licence",
    "in_corpus",
    "corpus_row",
    "sample",
    "sample_row",
]
MELD_COLUMNS = {
    "score": "meld_score",
    "n_tokens": "meld_tokens",
    "family": "meld_family",
    "task": "meld_task",
    "flag": "meld_flag",
    "written": "meld_written",
}
EDITLENS_COLUMNS = {"score": "editlens_score", "n_tokens": "editlens_tokens", "flag": "editlens_flag"}
GPTTRACE_COLUMNS = {
    "score": "gpttrace_score",
    "score_mean": "gpttrace_score_mean",
    "n_tokens": "gpttrace_tokens",
    "flag": "gpttrace_flag",
}
PANEL_KINDS = [
    ("placeboA_", "placebo_area"),
    ("placebo_", "placebo_venue"),
    ("otherplacebo_", "journal_placebo"),
    ("other_", "journal"),
    ("icml_", "icml"),
    ("sub3_", "subset3"),
    ("noq_", "noq"),
    ("nocq_", "nocq"),
]
NULLABLE_STRING_COLUMNS = ["reviewer", "rating_raw", "form", "decision", "sample", "meld_family", "meld_task"]
PAPER_COLUMNS = [
    "source",
    "venue_type",
    "journal",
    "field",
    "year",
    "paper_id",
    "title",
    "abstract",
    "decision",
    "withdrawn",
    "primary_area",
    "area_pred",
    "licence",
    "n_reviews",
    "corpus_row",
]


PANEL_FILES = [
    "panels",
    "panels_icml",
    "panels_noq",
    "panels_nocq",
    "panels_placebo",
    "panels_placebo_area",
    "panels_sub3",
    "panels_other",
    "panels_other_placebo",
]
PAPER_STRING_COLUMNS = [c for c in PAPER_COLUMNS if c not in ("year", "withdrawn", "n_reviews", "corpus_row")]


def journal_paper_files():
    return sorted(settings.OTHER_REVIEWS_DIR.glob("*_papers.parquet"))


def conference_reviews(data):
    ml = pd.read_parquet(
        data / "reviews.parquet",
        columns=[
            "conf",
            "year",
            "paper_id",
            "review_id",
            "reviewer",
            "rating",
            "confidence",
            "text",
            "form",
            "review_time",
            "withdrawn",
            "decision",
        ],
    )
    ml["reviewer"] = ml.reviewer.where(ml.reviewer.str.match(REVIEWER_LABEL, na=False))
    icml = pd.concat([pd.read_parquet(path) for path in sorted(data.glob("icml20*_reviews.parquet"))], ignore_index=True)
    icml = icml.drop(columns=["primary_area", "modified_time"]).assign(
        review_time=pd.to_datetime(icml.review_time, unit="ms"), withdrawn=False, decision="accept"
    )
    df = pd.concat([ml, icml], ignore_index=True)
    return df.assign(venue_type="conference", journal=df.conf, field="Machine learning", licence=OPENREVIEW_LICENCE)


def journal_licences():
    papers = [pd.read_parquet(path, columns=["paper_id", "licence"]) for path in journal_paper_files()]
    return pd.concat(papers).drop_duplicates("paper_id").set_index("paper_id").licence


def journal_reviews(data):
    df = pd.read_parquet(data / "other_reviews.parquet").drop(columns=["n_chars"])
    return df.assign(venue_type="journal", licence=df.paper_id.map(journal_licences()), withdrawn=False)


def sample_columns(data, df):
    df["sample"] = None
    df["sample_row"] = pd.array([pd.NA] * len(df), dtype="Int64")
    position = pd.Series(np.arange(len(df)), index=df.review_id)
    extra = []
    for name, file in SAMPLES.items():
        ids = pd.read_parquet(data / file, columns=["review_id"]).review_id
        assert df.loc[position.reindex(ids).dropna().astype(int), "sample"].isna().all()
        known = ids.isin(position.index)
        rows = position[ids[known]].to_numpy()
        df.loc[rows, "sample"] = name
        df.loc[rows, "sample_row"] = np.flatnonzero(known.to_numpy())
        if not known.all():
            missing = pd.read_parquet(data / file).reset_index(drop=True)
            missing = missing[~known.to_numpy()]
            extra.append(missing.assign(sample=name, sample_row=missing.index.to_numpy(), in_corpus=False))
    return pd.concat([df, *extra], ignore_index=True)


def all_reviews(data):
    df = pd.concat([conference_reviews(data), journal_reviews(data)], ignore_index=True)
    df = df.assign(in_corpus=True, corpus_row=np.arange(len(df)))
    df = sample_columns(data, df)
    extra = df.in_corpus.eq(False)
    df.loc[extra, ["venue_type", "journal", "field", "licence"]] = ["conference", "ICLR", "Machine learning", OPENREVIEW_LICENCE]
    df = df.rename(columns={"conf": "source"})
    raw = df.rating_raw.where(df.rating_raw.notna(), df.rating)
    df["rating_raw"] = raw.map(lambda v: None if pd.isna(v) else str(v))
    df["rating"] = pd.to_numeric(df.rating, errors="coerce")
    df["confidence"] = pd.to_numeric(df.confidence, errors="coerce")
    df["review_time"] = pd.to_datetime(df.review_time, errors="coerce", utc=True)
    df["corpus_row"] = df.corpus_row.astype("Int64")
    df["withdrawn"] = df.withdrawn.astype("boolean")
    assert df.review_id.is_unique
    df = df[REVIEW_COLUMNS]
    meld = pd.read_parquet(data / "meld_flags.parquet", columns=["review_id", *MELD_COLUMNS]).set_index("review_id")
    for column, name in MELD_COLUMNS.items():
        df[name] = df.review_id.map(meld[column])
    return df.astype({**dict.fromkeys(NULLABLE_STRING_COLUMNS, "string"), "meld_flag": "boolean", "meld_written": "boolean"})


def icml_papers(data):
    rows = []
    for path in sorted(data.glob("icml20*_raw.jsonl")):
        year = int(path.name[4:8])
        for submission in read_jsonl(path):
            content = submission["content"]
            rows.append(
                dict(
                    source="ICML",
                    year=year,
                    paper_id=submission["id"],
                    title=content.get("title", {}).get("value"),
                    abstract=content.get("abstract", {}).get("value"),
                    primary_area=content.get("primary_area", {}).get("value"),
                    decision="accept",
                    withdrawn=False,
                )
            )
    return pd.DataFrame(rows)


def all_papers(data, reviews):
    ml = pd.read_parquet(
        data / "papers.parquet",
        columns=["conf", "year", "paper_id", "title", "abstract", "decision", "withdrawn", "primary_area"],
    ).rename(columns={"conf": "source"})
    ml = ml.merge(pd.read_parquet(data / "paper_areas.parquet", columns=["paper_id", "area_pred"]), on="paper_id", how="left")
    conference = pd.concat([ml, icml_papers(data)], ignore_index=True)
    conference = conference.assign(
        venue_type="conference", journal=conference.source, field="Machine learning", licence=OPENREVIEW_LICENCE
    )
    journal = pd.concat(
        [pd.read_parquet(path).drop(columns=["n_reviews", "n_split"], errors="ignore") for path in journal_paper_files()],
        ignore_index=True,
    ).rename(columns={"conf": "source"})
    df = pd.concat([conference, journal.assign(venue_type="journal", withdrawn=False)], ignore_index=True)
    df["n_reviews"] = df.paper_id.map(reviews.groupby("paper_id").size()).fillna(0).astype(int)
    df["corpus_row"] = np.arange(len(df))
    df["withdrawn"] = df.withdrawn.astype("boolean")
    return df.reindex(columns=PAPER_COLUMNS).astype(dict.fromkeys(PAPER_STRING_COLUMNS, "string"))


def panel_kind(panel_id):
    return next((kind for prefix, kind in PANEL_KINDS if panel_id.startswith(prefix)), "paper")


def panel_row(row, kind):
    return dict(
        panel_id=row["panel_id"],
        kind=kind,
        review_ids=row["review_ids"],
        targets=row.get("targets", []) if "error" not in row else None,
        clusters=row.get("clusters") if "error" not in row else None,
        error=row.get("error"),
    )


def markdown_table(df):
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = [
        "| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in row) + " |" for row in df.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


def raw_variant(data, name, file):
    variant = pd.read_parquet(data / file).reset_index(drop=True).rename(columns={"conf": "source"})
    return variant.assign(variant=name, base_review_id=variant.review_id.str.removeprefix(f"{name}_"), sample_row=variant.index)


def with_kept_claims(config, part):
    columns = ["review_id", "claims", "claims_error"]
    folder = settings.FINAL_DIR / config
    if not folder.exists():
        return part.assign(claims=None, claims_error=None)
    kept = ds.dataset(folder, format="parquet").to_table(columns=columns, filter=ds.field("review_id").isin(list(part.review_id)))
    return part.join(kept.to_pandas().set_index("review_id"), on="review_id")


def with_new_claims(part, path):
    review_ids = set(part.review_id)
    rows = merge_rows([path], review_key, lambda row: row["review_id"] in review_ids) if path.exists() else []
    index = [row["review_id"] for row in rows]
    claims = pd.Series([row.get("claims") for row in rows], index=index, dtype=object)
    errors = pd.Series([row.get("error") for row in rows], index=index, dtype=object)
    touched = part.review_id.isin(index)
    part.loc[touched, "claims"] = part.review_id[touched].map(claims)
    part.loc[touched, "claims_error"] = part.review_id[touched].map(errors)
    return part.astype({"claims_error": "string"})


def with_editlens(part):
    path = settings.RAW_DIR / "editlens_flags.parquet"
    if not path.exists():
        return part
    flags = pd.read_parquet(path, columns=["review_id", *EDITLENS_COLUMNS]).set_index("review_id")
    for column, name in EDITLENS_COLUMNS.items():
        part[name] = part.review_id.map(flags[column])
    return part.astype({"editlens_tokens": "Int64", "editlens_flag": "boolean"})


def with_gpttrace(part):
    path = settings.RAW_DIR / "gpttrace_flags.parquet"
    if not path.exists():
        return part
    flags = pd.read_parquet(path, columns=["review_id", *GPTTRACE_COLUMNS]).set_index("review_id")
    for column, name in GPTTRACE_COLUMNS.items():
        part[name] = part.review_id.map(flags[column])
    return part.astype({"gpttrace_tokens": "Int64", "gpttrace_flag": "boolean"})


def with_raw_marks(part):
    ids = pd.read_parquet(settings.RAW_DIR / SAMPLES["main"], columns=["review_id"]).review_id
    new = ids[ids.isin(part.review_id[part["sample"].isna()])]
    if len(new):
        main = ds.dataset(settings.FINAL_DIR / "reviews", format="parquet").to_table(
            columns=["sample_row"], filter=ds.field("sample") == "main"
        )
        row = pd.Series(main.column("sample_row").to_pandas().max() + 1 + np.arange(len(new)), index=new.to_numpy())
        found = part.review_id.isin(row.index)
        part.loc[found, "sample"] = "main"
        part.loc[found, "sample_row"] = part.review_id[found].map(row)
    meld = pd.read_parquet(settings.RAW_DIR / "meld_flags.parquet", columns=["review_id", *MELD_COLUMNS]).set_index("review_id")
    for column, name in MELD_COLUMNS.items():
        part[name] = part.review_id.map(meld[column])
    return part.astype({"meld_family": "string", "meld_task": "string", "meld_flag": "boolean", "meld_written": "boolean"})


def built_panel_rows(kind):
    return raw_panel_rows(settings.RAW_DIR, kind) if kind == "paper" else final_panel_rows(kind)


def final_panel_rows(kind):
    path = settings.FINAL_DIR / "panels" / f"{kind}.parquet"
    return pq.read_table(path).to_pylist() if path.exists() else []


def raw_panel_rows(data, kind):
    kept = {panel_key(row): row for row in final_panel_rows(kind)}
    blank = dict(kind=kind, targets=None, clusters=None, error=None)
    panels = [
        panel for name in PANEL_FILES for panel in read_jsonl(data / f"{name}.jsonl") if panel_kind(panel["panel_id"]) == kind
    ]
    return [kept.get(panel_key(panel), {**panel, **blank}) for panel in panels]


def new_panel_rows(path, kind):
    if not path.exists():
        return []
    return [panel_row(row, kind) for row in merge_rows([path], panel_key, lambda row: panel_kind(row["panel_id"]) == kind)]


def write_final(model, review_parts, variants, papers, panel_rows):
    out = settings.FINAL_DIR
    for folder in ["reviews", "papers", "panels", "variants"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    claims_path = settings.WORK_DIR / f"claims_{model}.jsonl"
    summary = []
    for source, part in review_parts:
        part = with_gpttrace(with_editlens(with_new_claims(part, claims_path)))
        part.to_parquet(out / "reviews" / f"{source}.parquet", index=False, row_group_size=settings.ROW_GROUP_ROWS)
        summary.append(
            dict(
                source=source,
                years=f"{part.year.min()}–{part.year.max()}",
                papers=int(part.paper_id.nunique()),
                reviews=len(part),
                in_sample=int(part["sample"].notna().sum()),
                with_claims=int(part.claims.notna().sum()),
                with_meld=int(part.meld_score.notna().sum()),
                with_editlens=int(part.editlens_score.notna().sum()),
                with_gpttrace=int(part.gpttrace_score.notna().sum()),
            )
        )
        print(summary[-1], flush=True)
    for name, variant in variants:
        with_new_claims(variant, claims_path).to_parquet(
            out / "variants" / f"{name}.parquet", index=False, row_group_size=settings.ROW_GROUP_ROWS
        )
        print(name, len(variant), flush=True)
    papers.to_parquet(out / "papers" / "papers.parquet", index=False)
    panel_counts = []
    for kind in [kind for _, kind in PANEL_KINDS] + ["paper"]:
        rows = {row["panel_id"]: row for row in panel_rows(kind)}
        rows.update({row["panel_id"]: row for row in new_panel_rows(settings.WORK_DIR / f"clusters_{model}.jsonl", kind)})
        pd.DataFrame(list(rows.values())).astype({"error": "string"}).to_parquet(out / "panels" / f"{kind}.parquet", index=False)
        panel_counts.append(dict(kind=kind, panels=sum(row["clusters"] is not None for row in rows.values())))
        print(kind, len(rows), flush=True)
    meld = pd.read_csv(settings.REPORTS_DIR / "meld_summary.csv", index_col="name").value
    editlens = pd.read_csv(settings.REPORTS_DIR / "editlens_summary.csv", index_col="name").value
    gpttrace = pd.read_csv(settings.REPORTS_DIR / "gpttrace_summary.csv", index_col="name").value
    card = render(
        "dataset_card",
        sources=markdown_table(pd.DataFrame(summary)),
        panels=markdown_table(pd.DataFrame(panel_counts)),
        n_reviews=f"{sum(row['reviews'] for row in summary):,}",
        n_papers=f"{len(papers):,}",
        n_panels=f"{sum(p['panels'] for p in panel_counts):,}",
        threshold=f"{meld.threshold:.4f}",
        calibration_reviews=f"{int(meld.calibration_reviews):,}",
        calibration_last_year=settings.MELD_CALIBRATION_LAST_YEAR,
        quantile=settings.MELD_FLAG_QUANTILE,
        meld_repo=settings.MELD_REPO,
        meld_revision=settings.MELD_REVISION,
        editlens_threshold=f"{editlens.threshold:.4f}",
        editlens_repo=settings.EDITLENS_REPO,
        editlens_revision=settings.EDITLENS_REVISION,
        editlens_base_repo=settings.EDITLENS_BASE_REPO,
        editlens_max_length=f"{settings.EDITLENS_MAX_LENGTH:,}",
        gpttrace_threshold=f"{gpttrace.threshold:.2f}",
        gpttrace_repo=settings.GPTTRACE_REPO,
        gpttrace_revision=settings.GPTTRACE_REVISION,
        gpttrace_max_length=settings.GPTTRACE_MAX_LENGTH,
        model=model,
        route=settings.OPENROUTER_ROUTES[model],
        efforts=settings.REASONING_EFFORTS[model],
    )
    (settings.DATA_DIR / "README.md").write_text(card)
    (settings.DATA_DIR / "stats.json").write_text(json.dumps(dict(sources=summary, panels=panel_counts), indent=1))


def build(model=settings.DEFAULT_MODEL):
    final = settings.FINAL_DIR
    write_final(
        model,
        ((path.stem, with_raw_marks(pd.read_parquet(path))) for path in sorted((final / "reviews").glob("*.parquet"))),
        ((name, pd.read_parquet(final / "variants" / f"{name}.parquet")) for name in VARIANTS),
        pd.read_parquet(final / "papers" / "papers.parquet"),
        built_panel_rows,
    )


def from_raw(model=settings.DEFAULT_MODEL, data=str(settings.RAW_DIR)):
    data = Path(data)
    reviews = all_reviews(data)
    order = ["year", "paper_id", "review_id"]
    write_final(
        model,
        ((source, with_kept_claims("reviews", part.sort_values(order))) for source, part in reviews.groupby("source")),
        ((name, with_kept_claims("variants", raw_variant(data, name, file))) for name, file in VARIANTS.items()),
        all_papers(data, reviews[["paper_id"]]),
        partial(raw_panel_rows, data),
    )


def pull(raw=False):
    snapshot_download(
        settings.HF_DATASET_REPO,
        repo_type="dataset",
        revision=settings.HF_DATASET_REVISION,
        allow_patterns=settings.HF_PULL_PATTERNS + (settings.HF_RAW_PATTERNS if raw else []),
        local_dir=settings.DATA_DIR,
    )


def push():
    info = HfApi().upload_folder(
        repo_id=settings.HF_DATASET_REPO,
        folder_path=settings.DATA_DIR,
        repo_type="dataset",
        commit_message="Release reviews, papers, claims, panels, MELD, EditLens and GPTTrace scores",
        allow_patterns=settings.HF_PULL_PATTERNS + settings.HF_RAW_PATTERNS,
        ignore_patterns=settings.HF_PUSH_IGNORE,
        delete_patterns=["final/**", "gold/**", "additional/**"],
    )
    print(info.commit_url, info.oid)


if __name__ == "__main__":
    fire.Fire({"build": build, "from_raw": from_raw, "pull": pull, "push": push})
