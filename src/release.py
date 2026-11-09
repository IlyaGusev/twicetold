import json
import shutil
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

from jsonl import read_jsonl
from prompts import render
from settings import settings

OPENREVIEW_LICENCE = "OpenReview public review"
REVIEWER_LABEL = r"^(?:Reviewer|Referee)"
SAMPLES = {
    "main": "sample_reviews.parquet",
    "icml": "icml_sample.parquet",
    "journal": "other_pass1.parquet",
    "withdrawn": "wd_reviews.parquet",
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
PANEL_KINDS = [
    ("placeboA_", "placebo_area"),
    ("placebo_", "placebo_venue"),
    ("otherplacebo_", "journal_placebo"),
    ("other_", "journal"),
    ("icml_", "icml"),
    ("sub3_", "subset3"),
    ("wd_", "withdrawn"),
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


ARTIFACTS = {
    "meld": [
        "meld_scores.jsonl",
        "intel_reviews.parquet",
        f"ai_reference_{settings.AI_REFERENCE_MODEL}.jsonl",
        f"ai_polish_{settings.AI_REFERENCE_MODEL}.jsonl",
    ],
    "embeddings": ["review_emb.npy"],
    "openreview": ["icml2025_raw.jsonl", "icml2026_raw.jsonl", "iclr2027_raw.jsonl"],
    "areas": ["paper_areas.parquet"],
    "journals": ["other/stats.json"],
    "experiments": [
        "gold",
        "gold_audit",
        "gold_audit_written",
        "rewrite_test",
        "paper_reviews",
    ],
}
ARTIFACT_SKIPPED_DIRS = {"raw", "adjudicate2"}
PANEL_FILES = [
    "panels",
    "panels_icml",
    "panels_noq",
    "panels_nocq",
    "panels_wd",
    "panels_placebo",
    "panels_placebo_area",
    "panels_sub3",
    "panels_other",
    "panels_other_placebo",
]
ARTIFACTS["panel_files"] = [f"{name}.jsonl" for name in PANEL_FILES]
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


def extraction_of(path, review_ids):
    claims, errors = {}, {}
    for row in read_jsonl(path):
        if row["review_id"] in review_ids:
            if "claims" in row:
                claims[row["review_id"]] = row["claims"]
            else:
                errors[row["review_id"]] = row["error"]
    return claims, errors


def with_claims(part, claims_path):
    claims, errors = extraction_of(claims_path, set(part.review_id))
    return part.assign(claims=part.review_id.map(claims), claims_error=part.review_id.map(errors).astype("string"))


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


def panel_row(row, kind, current):
    return dict(
        panel_id=row["panel_id"],
        kind=kind,
        review_ids=row["review_ids"],
        current=(row["panel_id"], tuple(row["review_ids"])) in current,
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


def build(model=settings.DEFAULT_MODEL, data=str(settings.DATA_DIR), out=str(settings.RELEASE_DIR)):
    data, out = Path(data), Path(out)
    for folder in ["reviews", "papers", "panels", "variants"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    reviews = all_reviews(data)
    claims_path = data / f"claims_{model}.jsonl"
    summary = []
    for source, part in reviews.groupby("source"):
        part = with_claims(part, claims_path).sort_values(["year", "paper_id", "review_id"])
        part.to_parquet(out / "reviews" / f"{source}.parquet", index=False)
        summary.append(
            dict(
                source=source,
                years=f"{part.year.min()}–{part.year.max()}",
                papers=int(part.paper_id.nunique()),
                reviews=len(part),
                in_sample=int(part["sample"].notna().sum()),
                with_claims=int(part.claims.notna().sum()),
                with_meld=int(part.meld_score.notna().sum()),
            )
        )
        print(summary[-1], flush=True)
    for name, file in VARIANTS.items():
        variant = pd.read_parquet(data / file).reset_index(drop=True).rename(columns={"conf": "source"})
        variant = variant.assign(
            variant=name, base_review_id=variant.review_id.str.removeprefix(f"{name}_"), sample_row=variant.index
        )
        with_claims(variant, claims_path).to_parquet(out / "variants" / f"{name}.parquet", index=False)
        print(name, len(variant), flush=True)
    papers = all_papers(data, reviews[["paper_id"]])
    papers.to_parquet(out / "papers" / "papers.parquet", index=False)
    kinds = [kind for _, kind in PANEL_KINDS] + ["paper"]
    current = {(p["panel_id"], tuple(p["review_ids"])) for name in PANEL_FILES for p in read_jsonl(data / f"{name}.jsonl")}
    panel_counts = []
    for kind in kinds:
        rows = [
            panel_row(row, kind, current)
            for row in read_jsonl(data / f"clusters_{model}.jsonl")
            if panel_kind(row["panel_id"]) == kind
        ]
        pd.DataFrame(rows).astype({"error": "string"}).to_parquet(out / "panels" / f"{kind}.parquet", index=False)
        panel_counts.append(dict(kind=kind, panels=sum(row["error"] is None and row["current"] for row in rows)))
        print(kind, len(rows), flush=True)
    meld = pd.read_csv(settings.REPORTS_DIR / "meld_summary.csv", index_col="name").value
    card = render(
        "dataset_card",
        sources=markdown_table(pd.DataFrame(summary)),
        panels=markdown_table(pd.DataFrame(panel_counts)),
        n_reviews=f"{len(reviews):,}",
        n_papers=f"{len(papers):,}",
        n_panels=f"{sum(p['panels'] for p in panel_counts):,}",
        threshold=f"{meld.threshold:.4f}",
        calibration_reviews=f"{int(meld.calibration_reviews):,}",
        calibration_last_year=settings.MELD_CALIBRATION_LAST_YEAR,
        quantile=settings.MELD_FLAG_QUANTILE,
        meld_repo=settings.MELD_REPO,
        meld_revision=settings.MELD_REVISION,
        model=model,
        route=settings.OPENROUTER_ROUTES[model],
        efforts=settings.REASONING_EFFORTS[model],
    )
    (out / "README.md").write_text(card)
    (out / "stats.json").write_text(json.dumps(dict(sources=summary, panels=panel_counts), indent=1))


def artifact_files(data):
    for group, names in ARTIFACTS.items():
        for name in names:
            local = data / name
            if local.is_dir():
                files = [
                    path
                    for path in sorted(local.rglob("*"))
                    if path.is_file() and not ARTIFACT_SKIPPED_DIRS.intersection(path.relative_to(local).parts)
                ]
            else:
                files = [path for path in [local, local.with_name(local.name + ".md")] if path.exists()]
            for path in files:
                yield path, Path("artifacts") / group / path.relative_to(local.parent)


def artifacts(data=str(settings.DATA_DIR), out=str(settings.RELEASE_DIR)):
    data, out = Path(data), Path(out)
    for name in ARTIFACT_SKIPPED_DIRS:
        for path in (out / "artifacts").rglob(name):
            if path.is_dir():
                shutil.rmtree(path)
    for path, target in artifact_files(data):
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out / target)
        print(target, flush=True)


def pull(data=str(settings.DATA_DIR), repo=settings.HF_DATASET_REPO, revision=settings.HF_DATASET_REVISION):
    data = Path(data)
    root = Path(snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=["artifacts/**"]))
    for group, names in ARTIFACTS.items():
        for name in names:
            source = root / "artifacts" / group / Path(name).name
            target = data / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*ARTIFACT_SKIPPED_DIRS))
            else:
                for suffix in ["", ".md"]:
                    if source.with_name(source.name + suffix).exists():
                        shutil.copy2(source.with_name(source.name + suffix), target.with_name(target.name + suffix))
            print(target, flush=True)


def push(out=str(settings.RELEASE_DIR), repo=settings.HF_DATASET_REPO):
    info = HfApi().upload_folder(
        repo_id=repo,
        folder_path=out,
        repo_type="dataset",
        commit_message="Release reviews, papers, claims, panels and MELD scores",
        delete_patterns=["*.parquet", "artifacts/**"],
    )
    print(info.commit_url, info.oid)


def push_experiment(name, data=str(settings.DATA_DIR), repo=settings.HF_DATASET_REPO):
    info = HfApi().upload_folder(
        repo_id=repo,
        folder_path=str(Path(data) / name),
        path_in_repo=f"artifacts/experiments/{name}",
        repo_type="dataset",
        commit_message=f"Update artifacts/experiments/{name}",
        delete_patterns=["**"],
        ignore_patterns=[f"{skipped}/**" for skipped in ARTIFACT_SKIPPED_DIRS],
    )
    print(info.commit_url, info.oid)


if __name__ == "__main__":
    fire.Fire({"build": build, "artifacts": artifacts, "push": push, "push_experiment": push_experiment, "pull": pull})
