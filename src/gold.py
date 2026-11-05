import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract
from ai import paper_kind
from prompts import render
from settings import settings

CROSS_YEARS = (2018, 2022, 2024, 2026)
CROSS_PER_YEAR = 5
TEST_PER_CELL = {"same": 3, "cross": 2}
MAX_CLAIM_WORDS = 25
JUDGE_ATTEMPTS = 3
COUNCIL_RUNS = 2
GOLD_CLAIM_KEYS = ("claim", "category", "target", "specific", "support")
AUDIT_REVIEWS = 4
AUDIT_PER_STRATUM = {"same2023": 10, "flag": 15}
AUDIT_JOURNALS_PER_SOURCE = 3
AUDIT_FLAG_YEARS = (2024, 2026)
AUDIT_JOURNAL_FROM = 2024
AUDIT_WRITTEN_PANELS = 60
PAPER_KINDS = ("ml", "general")

GOLD_CLAIMS_SCHEMA = extract.strict_object(
    claims={
        "type": "array",
        "items": extract.strict_object(**extract.CLAIM_SCHEMA["properties"], support={"type": "integer", "enum": [0, 1, 2, 3]}),
    }
)
GOLD_GROUPS_SCHEMA = extract.strict_object(
    clusters={
        "type": "array",
        "items": extract.strict_object(problem={"type": "string"}, claim_ids={"type": "array", "items": {"type": "string"}}),
    }
)


def council_names(council):
    return ", ".join(settings.OPENROUTER_ROUTES[model] for model in council)


def by_model(path, key, field):
    rows = {}
    for row in extract.read_jsonl(path):
        if field in row:
            rows.setdefault(row["model"], {})[row[key]] = row[field]
    return rows


def merge_council(out, kind, key, tag, council):
    parts = [out / f"{kind}_{model}{tag}.jsonl" for model in council]
    extract.write_jsonl(out / f"{kind}_council.jsonl", extract.merge_rows(parts, lambda row: (row["model"], row[key])))
    for part in parts:
        part.unlink()


def sample(out, seed=7, data=str(settings.DATA_DIR), gold=str(settings.GOLD_DIR)):
    out, data = Path(out), Path(data)
    out.mkdir(parents=True, exist_ok=True)
    key = json.loads((Path(gold) / "blind_key.json").read_text())
    placebo = {p["panel_id"]: p for p in extract.read_jsonl(data / "panels_placebo.jsonl")}
    rng = np.random.default_rng(seed)
    panels = [dict(panel_id=code, review_ids=v["review_ids"], kind="same", year=v["year"]) for code, v in sorted(key.items())]
    for year in CROSS_YEARS:
        ids = sorted(p for p in placebo if p.startswith(f"placebo_ICLR_{year}_"))
        chosen = rng.choice(ids, CROSS_PER_YEAR, replace=False)
        panels += [dict(panel_id=p, review_ids=placebo[p]["review_ids"], kind="cross", year=year) for p in chosen]
    cells = {}
    for panel in panels:
        cells.setdefault((panel["kind"], panel["year"]), []).append(panel)
    for (kind, _), cell in cells.items():
        for i, panel in enumerate(cell):
            panel["split"] = "test" if i >= len(cell) - TEST_PER_CELL[kind] else "dev"
    extract.write_jsonl(out / "panels_tune.jsonl", panels)

    reviews = pd.read_parquet(data / "sample_reviews.parquet")
    blind_ids = {r for v in key.values() for r in v["review_ids"]}
    blind = reviews[reviews.review_id.isin(blind_ids)][["review_id", "text"]]
    panel_ids = {r for p in panels for r in p["review_ids"]}
    tune = pd.concat([blind, reviews])
    tune[tune.review_id.isin(panel_ids)].drop_duplicates("review_id").to_parquet(out / "tune_reviews.parquet")
    print(len(panels), "panels,", len(panel_ids), "reviews ->", out)


def flagged_split(review_ids, flag, rng):
    flagged = [r for r in review_ids if flag.get(r, False)]
    clean = [r for r in review_ids if r in flag and not flag[r]]
    if len(flagged) < 2 or len(clean) < 1:
        return None
    picked_flagged = list(rng.choice(flagged, 2, replace=False))
    picked_clean = list(rng.choice(clean, min(len(clean), AUDIT_REVIEWS - 2), replace=False))
    return [str(r) for r in picked_flagged + picked_clean]


def sample_audit(out=settings.AUDIT_DIR, seed=11, model=settings.DEFAULT_MODEL):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    flag = pd.read_parquet(settings.DATA_DIR / "meld_flags.parquet", columns=["review_id", "flag"]).set_index("review_id").flag
    flag = flag.to_dict()
    clusters = load_panels(settings.DATA_DIR / "panels.jsonl")
    iclr = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet", columns=["panel_id", "conf", "year"])
    iclr = iclr[iclr.conf == "ICLR"].sort_values("panel_id")
    panels = []
    same = iclr[(iclr.year == 2023) & iclr.panel_id.map(lambda p: len(clusters[p]) >= AUDIT_REVIEWS)]
    for p in same.sample(AUDIT_PER_STRATUM["same2023"], random_state=seed).itertuples():
        picked = [str(r) for r in rng.choice(clusters[p.panel_id], AUDIT_REVIEWS, replace=False)]
        panels.append(dict(source="ICLR", stratum="same2023", year=2023, paper_id=p.panel_id, review_ids=picked))
    late = iclr[iclr.year.between(*AUDIT_FLAG_YEARS)].sample(frac=1, random_state=seed)
    for p in late.itertuples():
        if sum(x["stratum"] == "flag" for x in panels) == AUDIT_PER_STRATUM["flag"]:
            break
        picked = flagged_split(clusters[p.panel_id], flag, rng)
        if picked and len(picked) == AUDIT_REVIEWS:
            panels.append(dict(source="ICLR", stratum="flag", year=int(p.year), paper_id=p.panel_id, review_ids=picked))
    journals = load_panels(settings.DATA_DIR / "panels_other.jsonl")
    meta = pd.read_parquet(settings.DATA_DIR / "other_reviews.parquet", columns=["review_id", "conf", "year"]).set_index(
        "review_id"
    )
    candidates = pd.DataFrame(
        [dict(panel_id=k, conf=meta.conf[v[0]], year=int(meta.year[v[0]])) for k, v in journals.items() if v[0] in meta.index]
    )
    candidates = candidates[candidates.year >= AUDIT_JOURNAL_FROM].sort_values("panel_id")
    for source, cell in candidates.groupby("conf"):
        chosen = 0
        for p in cell.sample(frac=1, random_state=seed).itertuples():
            if chosen == AUDIT_JOURNALS_PER_SOURCE:
                break
            picked = flagged_split(journals[p.panel_id], flag, rng)
            if picked:
                panels.append(dict(source=source, stratum="journal", year=p.year, paper_id=p.panel_id, review_ids=picked))
                chosen += 1
    write_audit(out, panels, model)


def sample_written(out=settings.AUDIT_WRITTEN_DIR, seed=12, model=settings.DEFAULT_MODEL, n=AUDIT_WRITTEN_PANELS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = pd.read_parquet(settings.DATA_DIR / "meld_flags.parquet", columns=["review_id", "flag", "written"]).set_index(
        "review_id"
    )
    clusters = load_panels(settings.DATA_DIR / "panels.jsonl")
    iclr = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet", columns=["panel_id", "conf", "year"])
    iclr = iclr[(iclr.conf == "ICLR") & iclr.year.between(*AUDIT_FLAG_YEARS)].sort_values("panel_id")
    panels = []
    for p in iclr.sample(frac=1, random_state=seed).itertuples():
        ids = [r for r in clusters[p.panel_id] if r in labels.index]
        written = [r for r in ids if labels.written[r]]
        clean = [r for r in ids if not labels.flag[r]]
        if len(written) >= 2 and len(clean) >= 2:
            picked = list(rng.choice(written, 2, replace=False)) + list(rng.choice(clean, 2, replace=False))
            panels.append(
                dict(source="ICLR", stratum="written", year=int(p.year), paper_id=p.panel_id, review_ids=[str(r) for r in picked])
            )
        if len(panels) == n:
            break
    write_audit(out, panels, model)


def write_audit(out, panels, model):
    for i, panel in enumerate(panels):
        panel.update(panel_id=f"A{i:02d}", kind="same", split="audit", paper_kind=paper_kind(panel["source"]))
    extract.write_jsonl(out / "panels_tune.jsonl", panels)
    texts = (
        pd.concat(
            [
                pd.read_parquet(settings.DATA_DIR / "sample_reviews.parquet", columns=["review_id", "text"]),
                pd.read_parquet(settings.DATA_DIR / "other_reviews.parquet", columns=["review_id", "text"]),
            ]
        )
        .drop_duplicates("review_id")
        .set_index("review_id")
        .text
    )
    order = [(r, p["paper_kind"]) for p in panels for r in p["review_ids"]]
    reviews = pd.DataFrame([dict(review_id=r, text=texts[r], paper_kind=k) for r, k in order]).drop_duplicates("review_id")
    reviews[["review_id", "text"]].to_parquet(out / "tune_reviews.parquet")
    for kind in PAPER_KINDS:
        reviews[reviews.paper_kind == kind][["review_id", "text"]].to_parquet(out / f"tune_reviews_{kind}.parquet")
        extract.write_jsonl(out / f"panels_tune_{kind}.jsonl", [p for p in panels if p["paper_kind"] == kind])
    ids = set(reviews.review_id)
    luna = extract.read_jsonl(settings.DATA_DIR / f"claims_{model}.jsonl")
    extract.write_jsonl(out / f"claims_{model}.jsonl", [row for row in luna if row["review_id"] in ids])
    print(pd.DataFrame(panels).groupby(["stratum", "source"]).size().to_string())
    print(len(panels), "panels,", len(reviews), "reviews ->", out)


def load_panels(path):
    return {p["panel_id"]: p["review_ids"] for p in extract.read_jsonl(path)}


def kind_files(out, stem, suffix):
    split = [(kind, out / f"{stem}_{kind}{suffix}") for kind in PAPER_KINDS]
    return [(kind, path) for kind, path in split if path.exists()] or [("ml", out / f"{stem}{suffix}")]


def split_council(out, kind, tag, council):
    merged = out / f"{kind}_council.jsonl"
    if not merged.exists():
        return
    rows = [row for row in extract.read_jsonl(merged) if not row.get("error")]
    for model in council:
        extract.write_jsonl(out / f"{kind}_{model}{tag}.jsonl", [row for row in rows if row["model"] == model])


def council_claims(out, council=settings.GOLD_COUNCIL, threads=16, budget=15.0):
    out = Path(out)
    split_council(out, "claims", "_tune", council)
    for _ in range(COUNCIL_RUNS):
        for model in council:
            for kind, path in kind_files(out, "tune_reviews", ".parquet"):
                extract.pass1(str(path), model=model, tag="_tune", threads=threads, budget=budget, paper_kind=kind, data=out)
    merge_council(out, "claims", "review_id", "_tune", council)


def drop_incomplete(out, missing):
    if not missing:
        return
    for path in [out / "panels_tune.jsonl", *[p for _, p in kind_files(out, "panels_tune", ".jsonl")]]:
        panels = list(extract.read_jsonl(path))
        kept = [p for p in panels if not missing & set(p["review_ids"])]
        extract.write_jsonl(path, kept)
        print(f"{path.name}: dropped {len(panels) - len(kept)} panels with missing claims")


def claim_inputs(out, council=settings.GOLD_COUNCIL):
    out = Path(out)
    folder = out / "adjudicate"
    folder.mkdir(exist_ok=True)
    texts = pd.read_parquet(out / "tune_reviews.parquet").set_index("review_id").text
    claims = by_model(out / "claims_council.jsonl", "review_id", "claims")
    drop_incomplete(out, {r for model in council for r in texts.index if r not in claims[model]})
    order = [r for p in extract.read_jsonl(out / "panels_tune.jsonl") for r in p["review_ids"]]
    items = [dict(review_id=r, text=texts[r], council={model: claims[model][r] for model in council}) for r in order]
    extract.write_jsonl(folder / "in.jsonl", items)
    (folder / "INSTRUCTIONS.md").write_text(render("gold_claims", council=council_names(council)))


def valid_claims(item, reply):
    return all(len(c["claim"].split()) <= MAX_CLAIM_WORDS for c in reply["claims"])


def valid_groups(item, reply):
    given = sorted(x for cluster in reply["clusters"] for x in cluster["claim_ids"])
    return given == sorted(c["id"] for c in item["claims"])


def judge(folder, schema, name, key, keys, valid, model, effort, threads, budget):
    instructions = (folder / "INSTRUCTIONS.md").read_text()
    out_path = folder / "out.jsonl"

    def one(item):
        for _ in range(JUDGE_ATTEMPTS):
            if extract.cost() > budget:
                return None
            reply = extract.call(instructions, json.dumps(item), schema, name, model=model, effort=effort)
            if "error" not in reply and valid(item, reply):
                return [{k: entry[k] for k in keys} for entry in reply[name]]
        return None

    items = list(extract.read_jsonl(folder / "in.jsonl"))
    done = {row[key]: row for row in extract.read_jsonl(out_path)} if out_path.exists() else {}
    todo = [item for item in items if item[key] not in done]
    with ThreadPoolExecutor(threads) as pool:
        replies = list(pool.map(one, todo))
    done.update({item[key]: {key: item[key], name: reply} for item, reply in zip(todo, replies) if reply is not None})
    extract.write_jsonl(out_path, [done[item[key]] for item in items if item[key] in done])
    failed = len(items) - len(done)
    print(f"{out_path}: {len(done)} of {len(items)} items, ${extract.cost():.2f}", flush=True)
    if failed:
        print(f"{failed} items failed or hit the budget; run again to retry them")


def judge_claims(out, model=settings.GOLD_JUDGE_MODEL, threads=8, budget=15.0):
    effort = extract.reasoning_efforts(model).pass1
    judge(
        Path(out) / "adjudicate",
        GOLD_CLAIMS_SCHEMA,
        "claims",
        "review_id",
        GOLD_CLAIM_KEYS,
        valid_claims,
        model,
        effort,
        threads,
        budget,
    )


def council_groups(out, council=settings.GOLD_COUNCIL, threads=12, budget=15.0):
    out = Path(out)
    split_council(out, "clusters", "_tune_gold", council)
    for _ in range(COUNCIL_RUNS):
        for model in council:
            for kind, path in kind_files(out, "panels_tune", ".jsonl"):
                extract.pass2(
                    str(path),
                    model=model,
                    tag="_tune_gold",
                    claims_file="adjudicate/out.jsonl",
                    threads=threads,
                    budget=budget,
                    paper_kind=kind,
                    data=out,
                )
    merge_council(out, "clusters", "panel_id", "_tune_gold", council)


def group_inputs(out, council=settings.GOLD_COUNCIL):
    out = Path(out)
    folder = out / "adjudicate2"
    folder.mkdir(exist_ok=True)
    gold = extract.load_claims(out / "adjudicate" / "out.jsonl")
    reviews = {r for p in extract.read_jsonl(out / "panels_tune.jsonl") for r in p["review_ids"]}
    drop_incomplete(out, reviews - set(gold))
    clusters = by_model(out / "clusters_council.jsonl", "panel_id", "clusters")
    items = []
    for panel in extract.read_jsonl(out / "panels_tune.jsonl"):
        claims = [
            dict(id=f"R{j}-{k}", claim=c["claim"], category=c["category"], specific=c["specific"])
            for j, review_id in enumerate(panel["review_ids"])
            for k, c in enumerate(gold[review_id])
        ]
        votes = {
            model: [dict(problem=c["problem"], claim_ids=c["claim_ids"]) for c in clusters[model][panel["panel_id"]]]
            for model in council
            if panel["panel_id"] in clusters[model]
        }
        items.append(
            dict(
                panel_id=panel["panel_id"],
                kind=panel["kind"],
                year=panel["year"],
                n_reviews=len(panel["review_ids"]),
                claims=claims,
                council=votes,
            )
        )
    extract.write_jsonl(folder / "in.jsonl", items)
    (folder / "INSTRUCTIONS.md").write_text(render("gold_groups", council=council_names(council)))


def judge_groups(out, model=settings.GOLD_JUDGE_MODEL, threads=8, budget=15.0):
    effort = extract.reasoning_efforts(model).pass2
    judge(
        Path(out) / "adjudicate2",
        GOLD_GROUPS_SCHEMA,
        "clusters",
        "panel_id",
        ("problem", "claim_ids"),
        valid_groups,
        model,
        effort,
        threads,
        budget,
    )


def model_groups(out=settings.AUDIT_DIR, model=settings.DEFAULT_MODEL, threads=16, budget=5.0):
    out = Path(out)
    for kind, path in kind_files(out, "panels_tune", ".jsonl"):
        extract.pass2(
            str(path), model=model, claims_file=f"claims_{model}.jsonl", threads=threads, budget=budget, paper_kind=kind, data=out
        )


def match_claims(
    model=settings.DEFAULT_MODEL,
    matcher=settings.GOLD_JUDGE_MODEL,
    threads=12,
    budget=15.0,
    data=str(settings.DATA_DIR),
    gold=str(settings.GOLD_DIR),
    folder=None,
):
    gold = Path(gold)
    folder = Path(folder) if folder else Path(data) / f"match_{model}"
    folder.mkdir(exist_ok=True)
    reference = extract.load_claims(gold / "adjudicate" / "out.jsonl")
    ours = extract.load_claims(Path(data) / f"claims_{model}.jsonl")
    order = [r for p in extract.read_jsonl(gold / "panels_tune.jsonl") for r in p["review_ids"]]
    claims = [dict(review_id=f"ref_{r}", claims=reference[r]) for r in order] + [dict(review_id=r, claims=ours[r]) for r in order]
    extract.write_jsonl(folder / "claims.jsonl", claims)
    extract.write_jsonl(folder / "panels.jsonl", [dict(panel_id=r, review_ids=[f"ref_{r}", r]) for r in order])
    extract.pass2(
        str(folder / "panels.jsonl"), model=matcher, claims_file="claims.jsonl", threads=threads, budget=budget, data=folder
    )


if __name__ == "__main__":
    fire.Fire(
        dict(
            sample=sample,
            sample_audit=sample_audit,
            sample_written=sample_written,
            model_groups=model_groups,
            council_claims=council_claims,
            claim_inputs=claim_inputs,
            judge_claims=judge_claims,
            council_groups=council_groups,
            group_inputs=group_inputs,
            judge_groups=judge_groups,
            match_claims=match_claims,
        )
    )
