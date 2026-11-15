import json
import urllib.request
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import extract
import hub
from jsonl import read_jsonl, write_jsonl
from settings import settings

SAME_PANELS = {
    "ICLR": {2018: 15, 2019: 15, 2020: 15, 2021: 15, 2022: 15, 2023: 30, 2024: 45, 2025: 45, 2026: 60, 2027: 60},
    "NeurIPS": {2024: 30, 2025: 30},
    "ICML": {2025: 30, 2026: 30},
}
JOURNAL_PANELS = 15
JOURNAL_YEARS = (2024, 2026)
CROSS_YEARS = (2018, 2022, 2024, 2026, 2027)
CROSS_PER_YEAR = 12
CROSS_PER_JOURNAL = 6
TEST_SHARE = 0.3
COUNCIL_RUNS = 2
PAPER_KINDS = ("ml", "general")


def merge_council(out, kind, key, tag, council):
    parts = [out / f"{kind}_{model}{tag}.jsonl" for model in council]
    write_jsonl(out / f"{kind}_council.jsonl", extract.merge_rows(parts, lambda row: (row["model"], row[key])))
    for part in parts:
        part.unlink()


def stratum(review_ids, flag, written):
    n_flagged = sum(flag[r] for r in review_ids)
    if n_flagged == 0:
        return "plain"
    if n_flagged < 2 or n_flagged == len(review_ids):
        return "other"
    return "written" if sum(written[r] for r in review_ids) >= 2 else "flagged"


def quotas(n, year, conf):
    if year < settings.FLAG_FROM:
        return {"plain": n}
    n_written = n // 4 if conf in settings.ML_VENUES else 0
    return {"plain": n - n // 2, "flagged": n // 2 - n_written, "written": n_written}


def panel_table(definitions, labels):
    conf, year, flag, written = (labels[c].to_dict() for c in ["conf", "year", "flag", "written"])
    return pd.DataFrame(
        [
            dict(panel_id=p, conf=conf[ids[0]], year=int(year[ids[0]]), stratum=stratum(ids, flag, written))
            for p, ids in sorted(definitions.items())
        ]
    )


def draw(rng, table, definitions, kind, cells):
    panels = []
    for conf, lo, hi, n in cells:
        cell = table[(table.conf == conf) & table.year.between(lo, hi)]
        for name, k in quotas(n, lo, conf).items() if kind == "same" else [("cross", n)]:
            pool = cell if kind == "cross" else cell[cell.stratum == name]
            chosen = rng.choice(pool.index.to_numpy(), k, replace=False)
            n_test = round(k * TEST_SHARE)
            for i, row in enumerate(table.loc[chosen].itertuples()):
                panels.append(
                    dict(
                        panel_id=row.panel_id,
                        review_ids=definitions[row.panel_id],
                        kind=kind,
                        conf=conf,
                        year=int(row.year),
                        stratum=name,
                        split="test" if i < n_test else "dev",
                        paper_kind=extract.paper_kind(conf),
                    )
                )
    return panels


def sample(out, seed=7):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = hub.meld_flags(["conf", "year", "flag", "written"])
    same = hub.panel_definitions(["paper", "icml", "journal"])
    cross = hub.panel_definitions(["placebo_area", "journal_placebo"])
    same_table, cross_table = panel_table(same, labels), panel_table(cross, labels)
    journals = sorted(set(same_table.conf) - set(settings.ML_VENUES))
    same_cells = [(conf, year, year, n) for conf, years in SAME_PANELS.items() for year, n in years.items()]
    same_cells += [(conf, *JOURNAL_YEARS, JOURNAL_PANELS) for conf in journals]
    cross_cells = [("ICLR", year, year, CROSS_PER_YEAR) for year in CROSS_YEARS]
    cross_cells += [(conf, *JOURNAL_YEARS, CROSS_PER_JOURNAL) for conf in journals]
    panels = draw(rng, same_table, same, "same", same_cells) + draw(rng, cross_table, cross, "cross", cross_cells)
    reviews = write_sample(out, panels)
    print(pd.DataFrame(panels).groupby(["kind", "conf", "stratum", "split"]).size().to_string())
    print(len(panels), "panels,", len(reviews), "reviews ->", out)


def write_sample(out, panels):
    write_jsonl(out / "panels_tune.jsonl", panels)
    texts = hub.texts({r for p in panels for r in p["review_ids"]})
    order = [(r, p["paper_kind"]) for p in panels for r in p["review_ids"]]
    reviews = pd.DataFrame([dict(review_id=r, text=texts[r], paper_kind=k) for r, k in order]).drop_duplicates("review_id")
    reviews[["review_id", "text"]].to_parquet(out / "tune_reviews.parquet")
    for kind in PAPER_KINDS:
        reviews[reviews.paper_kind == kind][["review_id", "text"]].to_parquet(out / f"tune_reviews_{kind}.parquet")
        write_jsonl(out / f"panels_tune_{kind}.jsonl", [p for p in panels if p["paper_kind"] == kind])
    return reviews


def kind_files(out, stem, suffix):
    split = [(kind, out / f"{stem}_{kind}{suffix}") for kind in PAPER_KINDS]
    return [(kind, path) for kind, path in split if path.exists()] or [("ml", out / f"{stem}{suffix}")]


def split_council(out, kind, tag, council):
    merged = out / f"{kind}_council.jsonl"
    if not merged.exists():
        return
    rows = [row for row in read_jsonl(merged) if not row.get("error")]
    for model in council:
        write_jsonl(out / f"{kind}_{model}{tag}.jsonl", [row for row in rows if row["model"] == model])


def model_claims(out, model=settings.DEFAULT_MODEL, threads=64, budget=5.0):
    out = Path(out)
    for kind, path in kind_files(out, "tune_reviews", ".parquet"):
        extract.pass1(str(path), model=model, threads=threads, budget=budget, paper_kind=kind, data=out)


def council_claims(out, council=settings.GOLD_COUNCIL, threads=16, budget=15.0):
    out = Path(out)
    split_council(out, "claims", "_tune", council)
    for _ in range(COUNCIL_RUNS):
        for model in council:
            for kind, path in kind_files(out, "tune_reviews", ".parquet"):
                extract.pass1(str(path), model=model, tag="_tune", threads=threads, budget=budget, paper_kind=kind, data=out)
    merge_council(out, "claims", "review_id", "_tune", council)


def drop_incomplete(out, model, council):
    reviews = set(pd.read_parquet(out / "tune_reviews.parquet", columns=["review_id"]).review_id)
    done = {(row["model"], row["review_id"]) for row in read_jsonl(out / "claims_council.jsonl") if "claims" in row}
    missing = reviews - set(extract.load_claims(out / f"claims_{model}.jsonl"))
    missing |= {r for member in council for r in reviews if (member, r) not in done}
    if not missing:
        return
    for path in [out / "panels_tune.jsonl", *[p for _, p in kind_files(out, "panels_tune", ".jsonl")]]:
        panels = list(read_jsonl(path))
        kept = [p for p in panels if not missing & set(p["review_ids"])]
        write_jsonl(path, kept)
        print(f"{path.name}: dropped {len(panels) - len(kept)} panels with missing claims")


def model_groups(out, model=settings.DEFAULT_MODEL, council=settings.GOLD_COUNCIL, threads=16, budget=5.0):
    out = Path(out)
    drop_incomplete(out, model, council)
    for kind, path in kind_files(out, "panels_tune", ".jsonl"):
        extract.pass2(
            str(path), model=model, claims_file=f"claims_{model}.jsonl", threads=threads, budget=budget, paper_kind=kind, data=out
        )


def council_groups(out, model=settings.DEFAULT_MODEL, council=settings.GOLD_COUNCIL, threads=12, budget=15.0):
    out = Path(out)
    drop_incomplete(out, model, council)
    split_council(out, "clusters", "_tune", council)
    for _ in range(COUNCIL_RUNS):
        for member in council:
            for kind, path in kind_files(out, "panels_tune", ".jsonl"):
                extract.pass2(
                    str(path),
                    model=member,
                    tag="_tune",
                    claims_file=f"claims_{model}.jsonl",
                    threads=threads,
                    budget=budget,
                    paper_kind=kind,
                    data=out,
                )
    merge_council(out, "clusters", "panel_id", "_tune", council)


def member_groups(out, council=settings.GOLD_COUNCIL, threads=12, budget=15.0):
    out = Path(out)
    rows = [row for row in read_jsonl(out / "claims_council.jsonl") if "claims" in row]
    for member in council:
        write_jsonl(out / f"claims_{member}.jsonl", [row for row in rows if row["model"] == member])
        for kind, path in kind_files(out, "panels_tune", ".jsonl"):
            extract.pass2(
                str(path),
                model=member,
                tag="_own",
                claims_file=f"claims_{member}.jsonl",
                threads=threads,
                budget=budget,
                paper_kind=kind,
                data=out,
            )


def kim_sample(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        settings.KIM_DATASET_REPO,
        "expert_annotation/eval-00000-of-00001.parquet",
        repo_type="dataset",
        revision=settings.KIM_DATASET_REVISION,
    )
    columns = ["paper_id", "annotator_source", "reviewer_id", "reviewer_type", "review_item_number", "review_item"]
    df = pd.read_parquet(path, columns=columns)
    df = df[(df.reviewer_type == "Human") & (df.annotator_source == "primary")]
    rows = [
        dict(review_id=f"kim_{paper}_{reviewer}", items=list(g.sort_values("review_item_number").review_item))
        for (paper, reviewer), g in df.groupby(["paper_id", "reviewer_id"], sort=True)
    ]
    write_jsonl(out / "kim_items.jsonl", rows)
    reviews = pd.DataFrame([dict(review_id=r["review_id"], text="\n\n".join(r["items"])) for r in rows])
    reviews.to_parquet(out / "kim_reviews.parquet", index=False)
    print(f"{len(rows)} human reviews, {sum(len(r['items']) for r in rows)} items")


def claimcheck_sample(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{settings.CLAIMCHECK_REPO}/{settings.CLAIMCHECK_REVISION}/data/texts/source"
    pairs = {}
    for name in ["main", "pilot"]:
        with urllib.request.urlopen(f"{base}/{name}.json") as response:
            pairs.update(json.load(response))
    rows = [
        dict(
            review_id=f"claimcheck_{key}", items=[w["Weakness span"] for w in pair["response"]["Weakness associated with claims"]]
        )
        for key, pair in sorted(pairs.items())
    ]
    write_jsonl(out / "claimcheck_items.jsonl", rows)
    reviews = pd.DataFrame([dict(review_id=f"claimcheck_{key}", text=pairs[key]["meta"]["review"]) for key in sorted(pairs)])
    reviews.to_parquet(out / "claimcheck_reviews.parquet", index=False)
    print(f"{len(rows)} reviews, {sum(len(r['items']) for r in rows)} weakness spans")


def external_claims(out, name, paper_kind, model=settings.DEFAULT_MODEL, threads=16, budget=5.0):
    out = Path(out)
    extract.pass1(
        str(out / f"{name}_reviews.parquet"),
        model=model,
        tag=f"_{name}",
        threads=threads,
        budget=budget,
        paper_kind=paper_kind,
        data=out,
    )


if __name__ == "__main__":
    fire.Fire(
        dict(
            sample=sample,
            model_claims=model_claims,
            council_claims=council_claims,
            model_groups=model_groups,
            council_groups=council_groups,
            member_groups=member_groups,
            kim_sample=kim_sample,
            claimcheck_sample=claimcheck_sample,
            external_claims=external_claims,
        )
    )
