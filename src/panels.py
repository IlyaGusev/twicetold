import re
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from jsonl import append_jsonl, read_jsonl, write_jsonl
from settings import settings

QUOTA = {
    ("ICLR", 2024): 3000,
    ("ICLR", 2025): 3000,
    ("ICLR", 2026): 6000,
    ("ICLR", 2027): 10000,
    **{("NeurIPS", y): 1000 for y in range(2021, 2026)},
}
MAX_REVIEWS = 5
MIN_ICML_REVIEW_CHARS = 200
NOQ_YEARS = (2024, 2025, 2026)
SUB3_FULL_YEARS = (2023, 2026)
ROBUST_PAPERS_PER_YEAR = 400
AREA_PLACEBOS_PER_YEAR = 300
QUESTIONS_RX = r"\[QUESTIONS\].*?(?=\[(?:LIMITATIONS|OTHER)\]|\Z)"
COPERNICUS_PREPRINT_JOURNAL = "EGUsphere"
NOQ_PREFIX = "noq_"
NOCQ_PREFIX = "nocq_"
NOCQ_YEAR = 2023
OTHER_TAG = "[OTHER]"


def by_paper_panels(reviews, prefix="", sort=False):
    for pid, g in reviews.groupby("paper_id", sort=sort):
        yield dict(panel_id=f"{prefix}{pid}", review_ids=g.review_id.tolist())


def review_ids_by_paper(reviews):
    return reviews.groupby("paper_id").review_id.apply(list)


def one_review_per_paper(rng, by_paper, paper_ids):
    return [str(rng.choice(by_paper[p])) for p in paper_ids]


def sub3_panel(rng, paper_id, review_ids):
    picked = rng.choice(review_ids, 3, replace=False)
    return dict(panel_id=f"sub3_{paper_id}", review_ids=[str(r) for r in picked])


def cap_reviews_per_paper(rng, reviews):
    shuffled = reviews.assign(rand=rng.random(len(reviews))).sort_values("rand")
    return shuffled.groupby("paper_id").head(MAX_REVIEWS).drop(columns="rand")


def texts_by_id(path, review_ids):
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=["review_id", "text"]):
        frame = batch.to_pandas()
        parts.append(frame[frame.review_id.isin(review_ids)])
    return pd.concat(parts).set_index("review_id").text


def ml(
    data=str(settings.RAW_DIR),
    out=str(settings.RAW_DIR),
    placebo_per_cell=300,
    seed=2026,
    current_corpus=str(settings.RAW_DIR),
):
    data, out, current_corpus = Path(data), Path(out), Path(current_corpus)
    rng = np.random.default_rng(seed)
    reviews = pd.read_parquet(data / "reviews.parquet", columns=["conf", "year", "paper_id", "review_id"])
    papers = pd.read_parquet(data / "papers.parquet", columns=["conf", "year", "paper_id", "withdrawn", "n_reviews", "decision"])
    eligible = papers[~papers.withdrawn & (papers.n_reviews >= settings.MIN_PANEL_REVIEWS) & papers.decision.ne("desk_reject")]

    chosen_per_cell = []
    for (conf, year), g in eligible.groupby(["conf", "year"]):
        quota = QUOTA.get((conf, year))
        paper_ids = g.paper_id.to_numpy()
        if quota is not None and quota < len(paper_ids):
            paper_ids = rng.choice(paper_ids, quota, replace=False)
        chosen_per_cell.append(pd.Series(paper_ids))
    chosen_ids = pd.concat(chosen_per_cell)

    sample = cap_reviews_per_paper(rng, reviews[reviews.paper_id.isin(chosen_ids)])
    sample = sample.sort_values(["conf", "year", "paper_id", "review_id"])
    sample["text"] = texts_by_id(data / "reviews.parquet", set(sample.review_id)).reindex(sample.review_id).to_numpy()
    sample.to_parquet(out / "sample_reviews.parquet")
    current_corpus_ids = pd.read_parquet(current_corpus / "reviews.parquet", columns=["review_id"]).review_id
    in_current_corpus = sample[sample.review_id.isin(current_corpus_ids)]
    panels = [p for p in by_paper_panels(in_current_corpus) if len(p["review_ids"]) >= settings.MIN_PANEL_REVIEWS]
    write_jsonl(out / "panels.jsonl", panels)

    def placebo():
        for (conf, year), g in sample.groupby(["conf", "year"]):
            by_paper = review_ids_by_paper(g)
            for k in range(placebo_per_cell):
                pids = rng.choice(by_paper.index, 4, replace=False)
                rids = one_review_per_paper(rng, by_paper, pids)
                yield dict(panel_id=f"placebo_{conf}_{year}_{k}", review_ids=rids)

    write_jsonl(out / "panels_placebo.jsonl", placebo())
    print(sample.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())
    print("total papers", sample.paper_id.nunique(), "reviews", len(sample), "chars", sample.text.str.len().sum() / 1e6, "M")


def icml(data=str(settings.RAW_DIR), out=str(settings.RAW_DIR), per_year=1500, seed=11):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    parts = []
    for year in (2025, 2026):
        year_reviews, paper_ids = icml_eligible(data, year)
        keep = rng.choice(paper_ids, per_year, replace=False)
        shuffled = year_reviews[year_reviews.paper_id.isin(keep)].sample(frac=1, random_state=0)
        parts.append(shuffled.groupby("paper_id").head(MAX_REVIEWS))
    sample = pd.concat(parts)
    sample[["conf", "year", "paper_id", "review_id", "rating", "confidence", "text"]].to_parquet(out / "icml_sample.parquet")
    write_jsonl(out / "panels_icml.jsonl", by_paper_panels(sample, "icml_", sort=True))
    print(
        sample.groupby("year")
        .agg(
            papers=("paper_id", "nunique"),
            reviews=("review_id", "size"),
            median_chars=("text", lambda s: int(s.str.len().median())),
        )
        .to_string()
    )


def icml_eligible(data, year):
    year_reviews = pd.read_parquet(data / f"icml{year}_reviews.parquet")
    year_reviews = year_reviews[year_reviews.text.str.len() > MIN_ICML_REVIEW_CHARS].drop_duplicates(["paper_id", "text"])
    reviews_per_paper = year_reviews.groupby("paper_id").size()
    return year_reviews, reviews_per_paper[reviews_per_paper >= settings.MIN_PANEL_REVIEWS].index


def extend(data=str(settings.RAW_DIR), seed=2028):
    data = Path(data)
    rng = np.random.default_rng(seed)
    sample = pd.read_parquet(data / "sample_reviews.parquet")
    reviews = pd.read_parquet(data / "reviews.parquet", columns=["conf", "year", "paper_id", "review_id"])
    papers = pd.read_parquet(data / "papers.parquet", columns=["conf", "year", "paper_id", "withdrawn", "n_reviews", "decision"])
    eligible = papers[~papers.withdrawn & (papers.n_reviews >= settings.MIN_PANEL_REVIEWS) & papers.decision.ne("desk_reject")]
    in_quota_cell = pd.Series(list(zip(eligible.conf, eligible.year)), index=eligible.index).isin(QUOTA)
    new_ids = set(eligible.paper_id[in_quota_cell]) - set(sample.paper_id)
    added = cap_reviews_per_paper(rng, reviews[reviews.paper_id.isin(new_ids)])
    added = added.sort_values(["conf", "year", "paper_id", "review_id"])
    added["text"] = texts_by_id(data / "reviews.parquet", set(added.review_id)).reindex(added.review_id).to_numpy()
    pd.concat([sample, added], ignore_index=True).to_parquet(data / "sample_reviews.parquet")
    append_jsonl(data / "panels.jsonl", (p for p in by_paper_panels(added) if len(p["review_ids"]) >= settings.MIN_PANEL_REVIEWS))
    print(added.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())

    icml_sample = pd.read_parquet(data / "icml_sample.parquet")
    parts = []
    for year in (2025, 2026):
        year_reviews, paper_ids = icml_eligible(data, year)
        new = year_reviews[year_reviews.paper_id.isin(set(paper_ids) - set(icml_sample.paper_id))]
        parts.append(new.sample(frac=1, random_state=0).groupby("paper_id").head(MAX_REVIEWS))
    icml_added = pd.concat(parts)[icml_sample.columns]
    pd.concat([icml_sample, icml_added], ignore_index=True).to_parquet(data / "icml_sample.parquet")
    append_jsonl(data / "panels_icml.jsonl", by_paper_panels(icml_added, "icml_", sort=True))
    print(icml_added.groupby("year").agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())


def new_cell(conf="ICLR", year=2027, data=str(settings.RAW_DIR), placebo_per_cell=300, seed=2029):
    data = Path(data)
    rng = np.random.default_rng(seed)
    sample = pd.read_parquet(data / "sample_reviews.parquet")
    if ((sample.conf == conf) & (sample.year == year)).any():
        raise SystemExit(f"{conf} {year} is already in the sample")
    cell = [("conf", "==", conf), ("year", "==", year)]
    reviews = pd.read_parquet(data / "reviews.parquet", columns=["conf", "year", "paper_id", "review_id"], filters=cell)
    papers = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "withdrawn", "n_reviews", "decision"], filters=cell)
    eligible = papers[~papers.withdrawn & (papers.n_reviews >= settings.MIN_PANEL_REVIEWS) & papers.decision.ne("desk_reject")]
    chosen = rng.choice(np.sort(eligible.paper_id.to_numpy()), QUOTA[(conf, year)], replace=False)
    added = cap_reviews_per_paper(rng, reviews[reviews.paper_id.isin(chosen)])
    added = added.sort_values(["conf", "year", "paper_id", "review_id"])
    added["text"] = texts_by_id(data / "reviews.parquet", set(added.review_id)).reindex(added.review_id).to_numpy()
    pd.concat([sample, added], ignore_index=True).to_parquet(data / "sample_reviews.parquet")
    append_jsonl(data / "panels.jsonl", (p for p in by_paper_panels(added) if len(p["review_ids"]) >= settings.MIN_PANEL_REVIEWS))
    by_paper = review_ids_by_paper(added)
    append_jsonl(
        data / "panels_placebo.jsonl",
        (
            dict(
                panel_id=f"placebo_{conf}_{year}_{k}",
                review_ids=one_review_per_paper(rng, by_paper, rng.choice(by_paper.index, 4, replace=False)),
            )
            for k in range(placebo_per_cell)
        ),
    )
    print(conf, year, "eligible", len(eligible), "papers", added.paper_id.nunique(), "reviews", len(added))


def preprint_shares(reviews):
    journal_of_paper = reviews.groupby(["conf", "year", "paper_id"]).journal.first()
    is_preprint = journal_of_paper.eq(COPERNICUS_PREPRINT_JOURNAL)
    return is_preprint.groupby(level=["conf", "year"]).agg(papers="size", preprints="sum", preprint_share="mean").reset_index()


def other(
    data=str(settings.RAW_DIR),
    out=str(settings.RAW_DIR),
    placebo_per_cell=150,
    seed=2027,
    reports=str(settings.REPORTS_DIR),
):
    data, out, reports = Path(data), Path(out), Path(reports)
    rng = np.random.default_rng(seed)
    reviews = cap_reviews_per_paper(rng, pd.read_parquet(data / "other_reviews.parquet"))
    reviews = reviews.sort_values(["conf", "year", "paper_id", "review_id"])
    reviews[["review_id", "text"]].to_parquet(out / "other_pass1.parquet")
    write_jsonl(out / "panels_other.jsonl", by_paper_panels(reviews, "other_"))

    def placebo():
        for (conf, year), g in reviews.groupby(["conf", "year"]):
            by_paper = g.groupby("paper_id").agg(journal=("journal", "first"), rids=("review_id", list))
            papers_per_journal = by_paper.journal[by_paper.journal != COPERNICUS_PREPRINT_JOURNAL].value_counts()
            papers_per_journal = papers_per_journal[papers_per_journal >= 3]
            if papers_per_journal.empty:
                continue
            weights = papers_per_journal / papers_per_journal.sum()
            for k in range(placebo_per_cell):
                journal = rng.choice(papers_per_journal.index, p=weights)
                pids = rng.choice(by_paper.index[by_paper.journal == journal], 3, replace=False)
                rids = one_review_per_paper(rng, by_paper.rids, pids)
                yield dict(panel_id=f"otherplacebo_{conf}_{year}_{k}", review_ids=rids)

    write_jsonl(out / "panels_other_placebo.jsonl", placebo())
    preprint_shares(reviews).to_csv(reports / "other_placebo_preprints.csv", index=False)
    print(reviews.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())
    print("reviews", len(reviews), "chars", reviews.text.str.len().sum() / 1e6, "M")


def without_questions(reviews):
    return reviews.assign(
        review_id=NOQ_PREFIX + reviews.review_id,
        text=reviews.text.str.replace(QUESTIONS_RX, "", regex=True, flags=re.DOTALL),
    )


def robust(data=str(settings.RAW_DIR), out=str(settings.RAW_DIR), seed=1, full_seed=11):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    iclr = pd.read_parquet(data / "sample_reviews.parquet")
    iclr = iclr[iclr.conf == "ICLR"]
    areas = pd.read_parquet(data / "paper_areas.parquet")[["paper_id", "area_pred"]]
    with_area = iclr.merge(areas, on="paper_id", how="left")
    with_area["area_pred"] = with_area.area_pred.fillna("other")
    main_panel_reviews = {r for p in read_jsonl(data / "panels.jsonl") for r in p["review_ids"]}

    def placebo_area():
        for year, g in with_area.groupby("year"):
            by_paper = review_ids_by_paper(g)
            area_of = g.drop_duplicates("paper_id").set_index("paper_id").area_pred
            for k in range(AREA_PLACEBOS_PER_YEAR):
                area = area_of.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
                area_papers = area_of.index[area_of == area]
                if len(area_papers) < 3:
                    continue
                pids = rng.choice(area_papers, 3, replace=False)
                rids = one_review_per_paper(rng, by_paper, pids)
                yield dict(panel_id=f"placeboA_ICLR_{year}_{k}", area=area, review_ids=rids)

    def sub3():
        for _, g in with_area.groupby("year"):
            by_paper = review_ids_by_paper(g)
            big = by_paper[by_paper.str.len() >= 4]
            for pid in rng.choice(big.index, min(ROBUST_PAPERS_PER_YEAR, len(big)), replace=False):
                panel = sub3_panel(rng, pid, by_paper[pid])
                if main_panel_reviews.issuperset(panel["review_ids"]):
                    yield panel

    write_jsonl(out / "panels_placebo_area.jsonl", placebo_area())
    sub3_panels = list(sub3())

    noq_parts = []
    for year in NOQ_YEARS:
        year_reviews = with_area[with_area.year == year]
        pids = rng.choice(year_reviews.paper_id.unique(), ROBUST_PAPERS_PER_YEAR, replace=False)
        noq_parts.append(without_questions(year_reviews[year_reviews.paper_id.isin(pids)]))
    noq_panels = [p for part in noq_parts for p in by_paper_panels(part, NOQ_PREFIX, sort=True)]
    noq = pd.concat(noq_parts)
    original_chars = with_area[with_area.review_id.isin(noq.review_id.str.removeprefix(NOQ_PREFIX))].text.str.len().sum()
    print("noq reviews", len(noq), "chars dropped share", 1 - noq.text.str.len().sum() / original_chars)

    new_sub3 = full_sub3(iclr, np.random.default_rng(full_seed), main_panel_reviews, {p["panel_id"] for p in sub3_panels})
    write_jsonl(out / "panels_sub3.jsonl", sub3_panels + new_sub3)
    print("new sub3 panels", len(new_sub3))

    noq_full = without_questions(iclr[iclr.year.isin(NOQ_YEARS) & ~iclr.paper_id.isin(noq.paper_id)])
    pd.concat([noq, noq_full])[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(out / "noq_reviews.parquet")
    write_jsonl(out / "panels_noq.jsonl", noq_panels + list(by_paper_panels(noq_full, NOQ_PREFIX, sort=True)))
    print("new noq reviews", len(noq_full), "papers", noq_full.paper_id.nunique())


def robust_cell(year=2027, data=str(settings.RAW_DIR), seed=2030):
    data = Path(data)
    rng = np.random.default_rng(seed)
    iclr = pd.read_parquet(data / "sample_reviews.parquet")
    iclr = iclr[(iclr.conf == "ICLR") & (iclr.year == year)]
    if any(r["panel_id"].startswith(f"placeboA_ICLR_{year}_") for r in read_jsonl(data / "panels_placebo_area.jsonl")):
        raise SystemExit(f"ICLR {year} already has robustness panels")
    areas = pd.read_parquet(data / "paper_areas.parquet")[["paper_id", "area_pred"]]
    with_area = iclr.merge(areas, on="paper_id", how="left")
    with_area["area_pred"] = with_area.area_pred.fillna("other")
    by_paper = review_ids_by_paper(with_area)
    area_of = with_area.drop_duplicates("paper_id").set_index("paper_id").area_pred
    placebo_area = []
    for k in range(AREA_PLACEBOS_PER_YEAR):
        area = area_of.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        area_papers = area_of.index[area_of == area]
        if len(area_papers) >= 3:
            rids = one_review_per_paper(rng, by_paper, rng.choice(area_papers, 3, replace=False))
            placebo_area.append(dict(panel_id=f"placeboA_ICLR_{year}_{k}", area=area, review_ids=rids))
    append_jsonl(data / "panels_placebo_area.jsonl", placebo_area)
    main_panel_reviews = {r for p in read_jsonl(data / "panels.jsonl") for r in p["review_ids"]}
    sub3 = full_sub3(iclr.assign(year=year), rng, main_panel_reviews, set(), years=(year,))
    append_jsonl(data / "panels_sub3.jsonl", sub3)
    pids = rng.choice(np.sort(iclr.paper_id.unique()), ROBUST_PAPERS_PER_YEAR, replace=False)
    noq = without_questions(iclr[iclr.paper_id.isin(pids)])[["conf", "year", "paper_id", "review_id", "text"]]
    noq_all = pd.read_parquet(data / "noq_reviews.parquet")
    pd.concat([noq_all, noq], ignore_index=True).to_parquet(data / "noq_reviews.parquet")
    append_jsonl(data / "panels_noq.jsonl", by_paper_panels(noq, NOQ_PREFIX, sort=True))
    print("placebo_area", len(placebo_area), "sub3", len(sub3), "noq reviews", len(noq), "papers", noq.paper_id.nunique())


def full_sub3(iclr, rng, main_panel_reviews, done_sub3, years=SUB3_FULL_YEARS):
    new_sub3 = []
    for year in years:
        by_paper = review_ids_by_paper(iclr[iclr.year == year])
        for pid, rids in by_paper[by_paper.str.len() >= 4].items():
            if f"sub3_{pid}" not in done_sub3:
                panel = sub3_panel(rng, pid, rids)
                if main_panel_reviews.issuperset(panel["review_ids"]):
                    new_sub3.append(panel)
    return new_sub3


def nocq(data=str(settings.RAW_DIR), out=str(settings.RAW_DIR)):
    data, out = Path(data), Path(out)
    reviews = pd.read_parquet(data / "sample_reviews.parquet")
    reviews = reviews[(reviews.conf == "ICLR") & (reviews.year == NOCQ_YEAR)].set_index("review_id")
    if not reviews.text.str.count(re.escape(OTHER_TAG)).eq(2).all():
        raise SystemExit(f"every ICLR {NOCQ_YEAR} review must have two {OTHER_TAG} sections; the second is the clarity field")
    ids = set(reviews.index)
    panels = [p for p in read_jsonl(data / "panels.jsonl") if set(p["review_ids"]) <= ids]
    review_ids = [r for p in panels for r in p["review_ids"]]
    kept = reviews.loc[review_ids]
    kept = kept.assign(text=kept.text.str.rsplit(OTHER_TAG, n=1).str[0].str.rstrip()).reset_index()
    print(
        "nocq reviews",
        len(kept),
        "chars dropped share",
        1 - kept.text.str.len().sum() / reviews.loc[review_ids].text.str.len().sum(),
    )
    kept["review_id"] = NOCQ_PREFIX + kept.review_id
    kept[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(out / "nocq_reviews.parquet")
    write_jsonl(
        out / "panels_nocq.jsonl",
        (dict(panel_id=NOCQ_PREFIX + p["panel_id"], review_ids=[NOCQ_PREFIX + r for r in p["review_ids"]]) for p in panels),
    )
    print("nocq panels", len(panels))


def wd(data=str(settings.RAW_DIR), out=str(settings.RAW_DIR)):
    reviews = pd.read_parquet(Path(data) / "wd_reviews.parquet", columns=["paper_id", "review_id"])
    write_jsonl(Path(out) / "panels_wd.jsonl", by_paper_panels(reviews, "wd_", sort=True))


if __name__ == "__main__":
    fire.Fire(
        {
            "ml": ml,
            "icml": icml,
            "extend": extend,
            "new_cell": new_cell,
            "robust_cell": robust_cell,
            "other": other,
            "robust": robust,
            "nocq": nocq,
            "wd": wd,
        }
    )
