import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract
from prompts import render
from settings import settings

CROSS_YEARS = (2018, 2022, 2024, 2026)
CROSS_PER_YEAR = 5
TEST_PER_CELL = {"same": 3, "cross": 2}
MAX_CLAIM_WORDS = 25
JUDGE_ATTEMPTS = 3
COUNCIL_RUNS = 2
GOLD_CLAIM_KEYS = ("claim", "category", "target", "specific", "support")

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


def council_claims(out, council=settings.GOLD_COUNCIL, threads=16, budget=15.0):
    out = Path(out)
    for _ in range(COUNCIL_RUNS):
        for model in council:
            extract.pass1(str(out / "tune_reviews.parquet"), model=model, tag="_tune", threads=threads, budget=budget, data=out)
    merge_council(out, "claims", "review_id", "_tune", council)


def claim_inputs(out, council=settings.GOLD_COUNCIL):
    out = Path(out)
    folder = out / "adjudicate"
    folder.mkdir(exist_ok=True)
    texts = pd.read_parquet(out / "tune_reviews.parquet").set_index("review_id").text
    claims = by_model(out / "claims_council.jsonl", "review_id", "claims")
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
    for _ in range(COUNCIL_RUNS):
        for model in council:
            extract.pass2(
                str(out / "panels_tune.jsonl"),
                model=model,
                tag="_tune_gold",
                claims_file="adjudicate/out.jsonl",
                threads=threads,
                budget=budget,
                data=out,
            )
    merge_council(out, "clusters", "panel_id", "_tune_gold", council)


def group_inputs(out, council=settings.GOLD_COUNCIL):
    out = Path(out)
    folder = out / "adjudicate2"
    folder.mkdir(exist_ok=True)
    gold = extract.load_claims(out / "adjudicate" / "out.jsonl")
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


if __name__ == "__main__":
    fire.Fire(
        dict(
            sample=sample,
            council_claims=council_claims,
            claim_inputs=claim_inputs,
            judge_claims=judge_claims,
            council_groups=council_groups,
            group_inputs=group_inputs,
            judge_groups=judge_groups,
        )
    )
