import json
import re
import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import settings

QUOTA = {("ICLR", 2024): 3000, ("ICLR", 2025): 3000, ("ICLR", 2026): 6000, **{("NeurIPS", y): 1000 for y in range(2021, 2026)}}
MAX_REVIEWS = 5
MIN_REVIEWS = 3
MIN_ICML_REVIEW_CHARS = 200
NOQ_YEARS = (2024, 2025, 2026)
SUB3_FULL_YEARS = (2023, 2026)
ROBUST_PAPERS_PER_YEAR = 400
AREA_PLACEBOS_PER_YEAR = 300
QUESTIONS_RX = r"\[QUESTIONS\].*?(?=\[(?:LIMITATIONS|OTHER)\]|\Z)"
COPERNICUS_PREPRINT_JOURNAL = "EGUsphere"


def read_panels(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_panels(path, panels):
    with open(path, "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in panels)


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


def sol(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), years=(2023, 2026), per_year=400, seed=5):
    data, out = Path(data), Path(out)
    done = {p["panel_id"] for p in read_panels(data / "panels_sol.jsonl")}
    panels = {p["panel_id"]: p for p in read_panels(data / "panels.jsonl")}
    outcomes = pd.read_parquet(
        data / f"panel_outcomes_{settings.BLIND_SET_MODEL}.parquet", columns=["panel_id", "conf", "year", "n_claims"]
    )
    eligible = outcomes[(outcomes.conf == "ICLR") & outcomes.year.isin(np.atleast_1d(years)) & (outcomes.n_claims > 0)]
    pool = eligible[~eligible.panel_id.isin(done)]
    picked_ids = pool.groupby("year", group_keys=False).sample(n=per_year, random_state=seed).panel_id
    chosen = [panels[p] for p in picked_ids]
    write_panels(out / "panels_sol2.jsonl", chosen)
    review_ids = {r for p in chosen for r in p["review_ids"]}
    texts_by_id(data / "reviews.parquet", review_ids).reset_index().to_parquet(out / "sol2_reviews.parquet")
    print(f"{len(chosen)} panels, {len(review_ids)} reviews")


def ml(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), placebo_per_cell=300, seed=2026):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    reviews = pd.read_parquet(data / "reviews.parquet", columns=["conf", "year", "paper_id", "review_id"])
    papers = pd.read_parquet(data / "papers.parquet", columns=["conf", "year", "paper_id", "withdrawn", "n_reviews", "decision"])
    eligible = papers[~papers.withdrawn & (papers.n_reviews >= MIN_REVIEWS) & papers.decision.ne("desk_reject")]

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
    corpus_ids = pd.read_parquet(settings.DATA_DIR / "reviews.parquet", columns=["review_id"]).review_id
    in_corpus = sample[sample.review_id.isin(corpus_ids)]
    write_panels(out / "panels.jsonl", (p for p in by_paper_panels(in_corpus) if len(p["review_ids"]) >= MIN_REVIEWS))

    def placebo():
        for (conf, year), g in sample.groupby(["conf", "year"]):
            by_paper = review_ids_by_paper(g)
            for k in range(placebo_per_cell):
                pids = rng.choice(by_paper.index, 4, replace=False)
                rids = one_review_per_paper(rng, by_paper, pids)
                yield dict(panel_id=f"placebo_{conf}_{year}_{k}", review_ids=rids)

    write_panels(out / "panels_placebo.jsonl", placebo())
    print(sample.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())
    print("total papers", sample.paper_id.nunique(), "reviews", len(sample), "chars", sample.text.str.len().sum() / 1e6, "M")


def icml(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), per_year=1500, seed=11):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    parts = []
    for year in (2025, 2026):
        year_reviews = pd.read_parquet(data / f"icml{year}_reviews.parquet")
        year_reviews = year_reviews[year_reviews.text.str.len() > MIN_ICML_REVIEW_CHARS].drop_duplicates(["paper_id", "text"])
        reviews_per_paper = year_reviews.groupby("paper_id").size()
        keep = rng.choice(reviews_per_paper[reviews_per_paper >= MIN_REVIEWS].index, per_year, replace=False)
        shuffled = year_reviews[year_reviews.paper_id.isin(keep)].sample(frac=1, random_state=0)
        parts.append(shuffled.groupby("paper_id").head(MAX_REVIEWS))
    sample = pd.concat(parts)
    sample[["review_id", "text"]].to_parquet(out / "icml_pass1.parquet")
    sample[["conf", "year", "paper_id", "review_id", "rating", "confidence", "text"]].to_parquet(out / "icml_sample.parquet")
    write_panels(out / "panels_icml.jsonl", by_paper_panels(sample, "icml_", sort=True))
    print(
        sample.groupby("year")
        .agg(
            papers=("paper_id", "nunique"),
            reviews=("review_id", "size"),
            median_chars=("text", lambda s: int(s.str.len().median())),
        )
        .to_string()
    )


def other(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), placebo_per_cell=150, seed=2027):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    reviews = cap_reviews_per_paper(rng, pd.read_parquet(data / "other_reviews.parquet"))
    reviews = reviews.sort_values(["conf", "year", "paper_id", "review_id"])
    reviews[["review_id", "text"]].to_parquet(out / "other_pass1.parquet")
    write_panels(out / "panels_other.jsonl", by_paper_panels(reviews, "other_"))

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

    write_panels(out / "panels_other_placebo.jsonl", placebo())
    print(reviews.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())
    print("reviews", len(reviews), "chars", reviews.text.str.len().sum() / 1e6, "M")


def drop_questions(reviews):
    return reviews.assign(text=reviews.text.str.replace(QUESTIONS_RX, "", regex=True, flags=re.DOTALL))


def robust(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), seed=1, full_seed=11):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    iclr = pd.read_parquet(data / "sample_reviews.parquet")
    iclr = iclr[iclr.conf == "ICLR"]
    areas = pd.read_parquet(data / "paper_areas.parquet")[["paper_id", "area_final"]]
    with_area = iclr.merge(areas, on="paper_id", how="left")
    with_area["area_final"] = with_area.area_final.fillna("other")
    main_panel_reviews = {r for p in read_panels(data / "panels.jsonl") for r in p["review_ids"]}

    def placebo_area():
        for year, g in with_area.groupby("year"):
            by_paper = review_ids_by_paper(g)
            area_of = g.drop_duplicates("paper_id").set_index("paper_id").area_final
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

    write_panels(out / "panels_placebo_area.jsonl", placebo_area())
    write_panels(out / "panels_sub3.jsonl", sub3())

    noq_parts = []
    for year in NOQ_YEARS:
        year_reviews = with_area[with_area.year == year]
        pids = rng.choice(year_reviews.paper_id.unique(), ROBUST_PAPERS_PER_YEAR, replace=False)
        noq_parts.append(drop_questions(year_reviews[year_reviews.paper_id.isin(pids)]))
    write_panels(out / "panels_noq.jsonl", (p for part in noq_parts for p in by_paper_panels(part, "noq_", sort=True)))
    noq = pd.concat(noq_parts)
    noq[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(out / "noq_reviews.parquet")
    original_chars = with_area[with_area.review_id.isin(noq.review_id)].text.str.len().sum()
    print("noq reviews", len(noq), "chars dropped share", 1 - noq.text.str.len().sum() / original_chars)
    robust_full(iclr, out, np.random.default_rng(full_seed), main_panel_reviews)


def robust_full(iclr, out, rng, main_panel_reviews):
    done_sub3 = {p["panel_id"] for p in read_panels(out / "panels_sub3.jsonl")}
    new_sub3 = []
    for year in SUB3_FULL_YEARS:
        by_paper = review_ids_by_paper(iclr[iclr.year == year])
        for pid, rids in by_paper[by_paper.str.len() >= 4].items():
            if f"sub3_{pid}" not in done_sub3:
                panel = sub3_panel(rng, pid, rids)
                if main_panel_reviews.issuperset(panel["review_ids"]):
                    new_sub3.append(panel)
    write_panels(out / "panels_sub3_full.jsonl", new_sub3)
    print("new sub3 panels", len(new_sub3))

    done_noq = {p["panel_id"] for p in read_panels(out / "panels_noq.jsonl")}
    in_noq = iclr.paper_id.map(lambda p: f"noq_{p}" in done_noq)
    noq_full = drop_questions(iclr[iclr.year.isin(NOQ_YEARS) & ~in_noq])
    noq_full[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(out / "noq_full_reviews.parquet")
    write_panels(out / "panels_noq_full.jsonl", by_paper_panels(noq_full, "noq_", sort=True))
    print("new noq reviews", len(noq_full), "papers", noq_full.paper_id.nunique())


if __name__ == "__main__":
    fire.Fire({"ml": ml, "icml": icml, "other": other, "robust": robust, "sol": sol})
