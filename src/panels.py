import sys
import json
import re
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import settings

QUOTA = {("ICLR", 2024): 3000, ("ICLR", 2025): 3000, ("ICLR", 2026): 6000,
         **{("NeurIPS", y): 1000 for y in range(2021, 2026)}}
MAX_REVIEWS = 5
QUESTIONS_RX = r"\[QUESTIONS\].*?(?=\[[A-Z]+\]|\Z)"


def read_panels(path):
    return [json.loads(line) for line in open(path)]


def write_panels(path, panels):
    with open(path, "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in panels)


def by_paper_panels(reviews, prefix="", sort=False):
    for pid, g in reviews.groupby("paper_id", sort=sort):
        yield dict(panel_id=f"{prefix}{pid}", review_ids=g.review_id.tolist())


def texts(path, ids):
    parts = []
    for b in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=["review_id", "text"]):
        b = b.to_pandas()
        parts.append(b[b.review_id.isin(ids)])
    return pd.concat(parts).set_index("review_id").text


def sol(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), years=(2023, 2026), per_year=400, seed=5):
    data, out = Path(data), Path(out)
    done = {p["panel_id"] for p in read_panels(data / "panels_sol.jsonl")}
    panels = {p["panel_id"]: p for p in read_panels(data / "panels.jsonl")}
    meta = pd.read_parquet(data / "panel_outcomes_gpt-6-luna.parquet", columns=["panel_id", "conf", "year", "n_claims"])
    pool = meta[(meta.conf == "ICLR") & meta.year.isin(np.atleast_1d(years)) & (meta.n_claims > 0) & ~meta.panel_id.isin(done)]
    pick = pool.groupby("year", group_keys=False).sample(n=per_year, random_state=seed).panel_id
    chosen = [panels[p] for p in pick]
    write_panels(out / "panels_sol2.jsonl", chosen)
    ids = {r for p in chosen for r in p["review_ids"]}
    texts(data / "reviews.parquet", ids).reset_index().to_parquet(out / "sol2_reviews.parquet")
    print(f"{len(chosen)} panels, {len(ids)} reviews")


def ml(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), placebo_per_cell=300, seed=2026):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    reviews = pd.read_parquet(data / "reviews.parquet", columns=["conf", "year", "paper_id", "review_id"])
    papers = pd.read_parquet(data / "papers.parquet", columns=["conf", "year", "paper_id", "withdrawn", "n_reviews",
                                                               "decision"])
    ok = papers[~papers.withdrawn & (papers.n_reviews >= 3) & papers.decision.ne("desk_reject")]

    chosen = []
    for (conf, year), g in ok.groupby(["conf", "year"]):
        n = QUOTA.get((conf, year))
        ids = g.paper_id.to_numpy()
        if n is not None and n < len(ids):
            ids = rng.choice(ids, n, replace=False)
        chosen.append(pd.DataFrame(dict(conf=conf, year=year, paper_id=ids)))
    chosen = pd.concat(chosen)

    sel = reviews[reviews.paper_id.isin(chosen.paper_id)].copy()
    sel["rand"] = rng.random(len(sel))
    sel = sel.sort_values("rand").groupby("paper_id").head(MAX_REVIEWS).drop(columns="rand")
    sel = sel.sort_values(["conf", "year", "paper_id", "review_id"])
    sel["text"] = texts(data / "reviews.parquet", set(sel.review_id)).reindex(sel.review_id).to_numpy()
    sel.to_parquet(out / "sample_reviews.parquet")
    write_panels(out / "panels.jsonl", by_paper_panels(sel))

    def placebo():
        for (conf, year), g in sel.groupby(["conf", "year"]):
            by_paper = g.groupby("paper_id").review_id.apply(list)
            for k in range(placebo_per_cell):
                pids = rng.choice(by_paper.index, 4, replace=False)
                rids = [str(rng.choice(by_paper[p])) for p in pids]
                yield dict(panel_id=f"placebo_{conf}_{year}_{k}", review_ids=rids)

    write_panels(out / "panels_placebo.jsonl", placebo())
    print(sel.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())
    print("total papers", sel.paper_id.nunique(), "reviews", len(sel), "chars", sel.text.str.len().sum() / 1e6, "M")


def icml(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), per_year=1500, seed=11):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    parts = []
    for year in (2025, 2026):
        d = pd.read_parquet(data / f"icml{year}_reviews.parquet")
        d = d[d.text.str.len() > 200].drop_duplicates(["paper_id", "text"])
        sizes = d.groupby("paper_id").size()
        keep = rng.choice(sizes[sizes >= 3].index, per_year, replace=False)
        parts.append(d[d.paper_id.isin(keep)].sample(frac=1, random_state=0).groupby("paper_id").head(MAX_REVIEWS))
    d = pd.concat(parts)
    d[["review_id", "text"]].to_parquet(out / "icml_pass1.parquet")
    d[["conf", "year", "paper_id", "review_id", "rating", "confidence", "text"]].to_parquet(out / "icml_sample.parquet")
    write_panels(out / "panels_icml.jsonl", by_paper_panels(d, "icml_", sort=True))
    print(d.groupby("year").agg(papers=("paper_id", "nunique"), reviews=("review_id", "size"),
                                median_chars=("text", lambda s: int(s.str.len().median()))).to_string())


def other(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), placebo_per_cell=150, seed=2027):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    R = pd.read_parquet(data / "other_reviews.parquet")
    R["rand"] = rng.random(len(R))
    R = R.sort_values("rand").groupby("paper_id").head(MAX_REVIEWS).sort_values(["conf", "year", "paper_id", "review_id"])
    R[["review_id", "text"]].to_parquet(out / "other_pass1.parquet")
    write_panels(out / "panels_other.jsonl", by_paper_panels(R, "other_"))

    def placebo():
        for (conf, year), g in R.groupby(["conf", "year"]):
            by_paper = g.groupby("paper_id").agg(journal=("journal", "first"), rids=("review_id", list))
            journals = by_paper.journal.value_counts()
            journals = journals[journals >= 3].index
            if len(journals) == 0:
                continue
            for k in range(placebo_per_cell):
                j = rng.choice(journals, p=by_paper.journal.value_counts()[journals] / by_paper.journal.isin(journals).sum())
                pids = rng.choice(by_paper.index[by_paper.journal == j], 3, replace=False)
                rids = [str(rng.choice(by_paper.rids[p])) for p in pids]
                yield dict(panel_id=f"otherplacebo_{conf}_{year}_{k}", review_ids=rids)

    write_panels(out / "panels_other_placebo.jsonl", placebo())
    print(R.groupby(["conf", "year"]).agg(papers=("paper_id", "nunique"), reviews=("review_id", "size")).to_string())
    print("reviews", len(R), "chars", R.text.str.len().sum() / 1e6, "M")


def drop_questions(g):
    return g.assign(text=g.text.str.replace(QUESTIONS_RX, "", regex=True, flags=re.DOTALL))


def robust(data=str(settings.DATA_DIR), out=str(settings.DATA_DIR), seed=1, full_seed=11):
    data, out = Path(data), Path(out)
    rng = np.random.default_rng(seed)
    iclr = pd.read_parquet(data / "sample_reviews.parquet")
    iclr = iclr[iclr.conf == "ICLR"]
    areas = pd.read_parquet(data / "paper_areas.parquet")[["paper_id", "area_final"]]
    sel = iclr.merge(areas, on="paper_id", how="left")
    sel["area_final"] = sel.area_final.fillna("other")

    def placebo_area():
        for year, g in sel.groupby("year"):
            by_paper = g.groupby("paper_id").review_id.apply(list)
            area_of = g.drop_duplicates("paper_id").set_index("paper_id").area_final
            for k in range(300):
                area = area_of.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
                pool = area_of.index[area_of == area]
                if len(pool) < 3:
                    continue
                pids = rng.choice(pool, 3, replace=False)
                rids = [str(rng.choice(by_paper[p])) for p in pids]
                yield dict(panel_id=f"placeboA_ICLR_{year}_{k}", area=area, review_ids=rids)

    def sub3():
        for _, g in sel.groupby("year"):
            by_paper = g.groupby("paper_id").review_id.apply(list)
            big = by_paper[by_paper.str.len() >= 4]
            for pid in rng.choice(big.index, min(400, len(big)), replace=False):
                yield dict(panel_id=f"sub3_{pid}", review_ids=[str(r) for r in rng.choice(by_paper[pid], 3, replace=False)])

    write_panels(out / "panels_placebo_area.jsonl", placebo_area())
    write_panels(out / "panels_sub3.jsonl", sub3())

    noq = []
    for year in [2024, 2025, 2026]:
        g = sel[sel.year == year]
        pids = rng.choice(g.paper_id.unique(), 400, replace=False)
        noq.append(drop_questions(g[g.paper_id.isin(pids)]))
    write_panels(out / "panels_noq.jsonl", (p for g in noq for p in by_paper_panels(g, "noq_", sort=True)))
    noq = pd.concat(noq)
    noq[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(out / "noq_reviews.parquet")
    print("noq reviews", len(noq), "chars dropped share",
          1 - noq.text.str.len().sum() / sel[sel.review_id.isin(noq.review_id)].text.str.len().sum())
    robust_full(iclr, out, np.random.default_rng(full_seed))


def robust_full(sel, out, rng):
    done_sub3 = {p["panel_id"] for p in read_panels(out / "panels_sub3.jsonl")}
    sub3 = []
    for year in [2023, 2026]:
        by_paper = sel[sel.year == year].groupby("paper_id").review_id.apply(list)
        for pid, rids in by_paper[by_paper.str.len() >= 4].items():
            if f"sub3_{pid}" not in done_sub3:
                sub3.append(dict(panel_id=f"sub3_{pid}", review_ids=[str(r) for r in rng.choice(rids, 3, replace=False)]))
    write_panels(out / "panels_sub3_full.jsonl", sub3)
    print("new sub3 panels", len(sub3))

    done_noq = {p["panel_id"] for p in read_panels(out / "panels_noq.jsonl")}
    g = sel[sel.year.isin([2024, 2025, 2026]) & ~sel.paper_id.map(lambda p: f"noq_{p}" in done_noq)]
    g = drop_questions(g)
    g[["conf", "year", "paper_id", "review_id", "text"]].to_parquet(out / "noq_full_reviews.parquet")
    write_panels(out / "panels_noq_full.jsonl", by_paper_panels(g, "noq_", sort=True))
    print("new noq reviews", len(g), "papers", g.paper_id.nunique())


if __name__ == "__main__":
    fire.Fire({"ml": ml, "icml": icml, "other": other, "robust": robust, "sol": sol})
