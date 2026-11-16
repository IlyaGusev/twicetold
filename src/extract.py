import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache, partial
from pathlib import Path

import fire
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

import hub
from jsonl import read_jsonl
from prompts import render
from settings import settings

PASS1_PAPER_WORDING = {"ml": "machine-learning conference paper", "general": "scientific paper submitted to a journal"}
PASS2_PAPER_WORDING = {"ml": "machine-learning paper", "general": "scientific paper"}


def paper_kind(conf):
    return "ml" if conf in settings.ML_VENUES else "general"


CATEGORIES = [
    "novelty",
    "baselines/comparisons",
    "ablation",
    "experiments-scope",
    "statistical-rigor",
    "theory/correctness",
    "assumptions",
    "unsupported-claim",
    "weak-results",
    "clarity/presentation",
    "related-work",
    "reproducibility/details",
    "compute-cost/scalability",
    "generalization",
    "limitations/societal",
    "typo/formatting",
    "other",
]
TARGETS = ["method", "theory", "experiments", "results", "claims", "related-work", "presentation", "other"]


def strict_object(**properties):
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


CLAIM_SCHEMA = strict_object(
    quote={"type": "string"},
    claim={"type": "string"},
    category={"type": "string", "enum": CATEGORIES},
    target={"type": "string", "enum": TARGETS},
    specific={"type": "boolean"},
)
PASS1_SCHEMA = strict_object(claims={"type": "array", "items": CLAIM_SCHEMA})

TARGET_SCHEMA = strict_object(target_id={"type": "integer"}, element={"type": "string"})
CLUSTER_SCHEMA = strict_object(
    target_id={"type": "integer"}, problem={"type": "string"}, claim_ids={"type": "array", "items": {"type": "string"}}
)
PASS2_SCHEMA = strict_object(
    targets={"type": "array", "items": TARGET_SCHEMA}, clusters={"type": "array", "items": CLUSTER_SCHEMA}
)


JSON_TYPES = {"string": str, "integer": int, "boolean": bool}
QUOTE_CHARS = str.maketrans(
    {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "\u2013": "-", "\u2014": "-", "\\": None}
)

CLIENT_MAX_RETRIES = 6
CLIENT_TIMEOUT_SECONDS = 180
API_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 5
ERROR_REPR_CHARS = 300

usage_by_model = {}
usage_lock = threading.Lock()


@cache
def openai_client():
    return OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=CLIENT_MAX_RETRIES, timeout=CLIENT_TIMEOUT_SECONDS)


@cache
def openrouter_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
        max_retries=CLIENT_MAX_RETRIES,
        timeout=CLIENT_TIMEOUT_SECONDS,
    )


def empty_usage():
    return {"in": 0, "out": 0, "calls": 0}


def record(model, n_in, n_out):
    with usage_lock:
        usage = usage_by_model.setdefault(model, empty_usage())
        usage["in"] += n_in
        usage["out"] += n_out
        usage["calls"] += 1


def cost():
    total = 0.0
    with usage_lock:
        for model, usage in usage_by_model.items():
            price_in, price_out = settings.PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
            total += (usage["in"] * price_in + usage["out"] * price_out) / 1e6
    return total


def parse_object(text, schema=None):
    out = json.loads(text)
    if not isinstance(out, dict):
        raise ValueError(f"response is not a JSON object: {text[:80]}")
    if schema is not None and not conforms(out, schema):
        raise ValueError(f"response does not follow the schema: {text[:80]}")
    return out


def conforms(value, schema):
    if "enum" in schema:
        return value in schema["enum"]
    kind = schema["type"]
    if kind == "object":
        return (
            isinstance(value, dict)
            and all(key in value for key in schema.get("required", []))
            and all(key in schema["properties"] and conforms(item, schema["properties"][key]) for key, item in value.items())
        )
    if kind == "array":
        return isinstance(value, list) and all(conforms(item, schema["items"]) for item in value)
    if kind == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, JSON_TYPES[kind])


def openai_call(instructions, text, schema, name, model, effort):
    response = openai_client().responses.create(
        model=model,
        instructions=instructions,
        input=text,
        reasoning={"effort": effort},
        text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
    )
    record(model, response.usage.input_tokens, response.usage.output_tokens)
    return parse_object(response.output_text)


def openrouter_provider(model):
    provider = {"require_parameters": True}
    if model in settings.OPENROUTER_PROVIDERS:
        provider["only"] = [settings.OPENROUTER_PROVIDERS[model]]
    return provider


def openrouter_call(instructions, text, schema, name, model, effort):
    response = openrouter_client().chat.completions.create(
        model=settings.OPENROUTER_ROUTES[model],
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": text}],
        response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
        extra_body={"reasoning": {"effort": effort}, "provider": openrouter_provider(model)},
    )
    record(model, response.usage.prompt_tokens, response.usage.completion_tokens)
    return parse_object(response.choices[0].message.content, schema)


def normalize_quote(text):
    return " ".join(text.translate(QUOTE_CHARS).lower().split())


def quote_in_text(quote, text):
    return normalize_quote(quote) in normalize_quote(text)


def error_row(e):
    return {"error": repr(e)[:ERROR_REPR_CHARS]}


def call(instructions, text, schema, name, model, effort):
    for attempt in range(API_ATTEMPTS):
        try:
            if model in settings.OPENROUTER_ROUTES:
                return openrouter_call(instructions, text, schema, name, model, effort)
            return openai_call(instructions, text, schema, name, model, effort)
        except Exception as e:
            if attempt == API_ATTEMPTS - 1:
                return error_row(e)
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))


def load_claims(*paths):
    return {row["review_id"]: row["claims"] for path in paths if path.exists() for row in read_jsonl(path) if "claims" in row}


def merge_rows(paths, key, wanted=lambda row: True):
    kept = {}
    for path in paths:
        for row in read_jsonl(path):
            if wanted(row) and ("error" not in row or "error" in kept.get(key(row), {"error": None})):
                kept[key(row)] = row
    return list(kept.values())


def review_key(row):
    return row["review_id"]


def panel_key(row):
    return row["panel_id"], tuple(row["review_ids"])


def reasoning_efforts(model):
    if model not in settings.REASONING_EFFORTS:
        raise SystemExit(f"{model} has no entry in settings.REASONING_EFFORTS; each effort setting needs its own short name")
    return settings.REASONING_EFFORTS[model]


def done_ids(path, key, effort):
    if not path.exists():
        return set()
    done = set()
    for row in read_jsonl(path):
        if row.get("effort", effort) != effort:
            raise SystemExit(f"{path} has rows made at effort {row['effort']}, not {effort}; use a new short name")
        if "error" not in row:
            done.add(key(row))
    return done


def run(items, worker, out_path, done, key, model, effort, threads, budget):
    if model not in settings.PRICES_USD_PER_MTOK:
        raise SystemExit(f"{model} has no entry in settings.PRICES_USD_PER_MTOK, so the budget cap would not count its calls")
    done = done | done_ids(out_path, key, effort)
    todo = [item for item in items if key(item) not in done]
    print(f"{model}: {len(done)} done, {len(todo)} to do -> {out_path}", flush=True)

    def capped(item):
        return None if cost() > budget else worker(item)

    pool = ThreadPoolExecutor(threads)
    failed = 0
    try:
        with open(out_path, "a") as f:
            bar = tqdm(as_completed([pool.submit(capped, item) for item in todo]), total=len(todo), desc=out_path.stem)
            for future in bar:
                row = future.result()
                if row is not None:
                    failed += "error" in row
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                bar.set_postfix(cost=f"${cost():.2f}")
    finally:
        pool.shutdown(cancel_futures=True)
    if cost() > budget:
        print("budget cap reached", flush=True)
    if failed:
        print(f"{failed} items failed", flush=True)
    usage = usage_by_model.get(model, empty_usage())
    print(
        f"finished: {usage['calls']} calls, {usage['in'] / 1e6:.2f}M in, {usage['out'] / 1e6:.2f}M out, ${cost():.2f}", flush=True
    )


def pass1_instructions(paper_kind):
    return render("pass1", paper=PASS1_PAPER_WORDING[paper_kind], categories=", ".join(CATEGORIES), targets=", ".join(TARGETS))


def pass1_worker(item, model, instructions, effort):
    out = call(instructions, item["text"], PASS1_SCHEMA, "claims", model=model, effort=effort)
    return dict(review_id=item["review_id"], model=model, effort=effort, **out)


def pass2_worker(item, model, instructions, effort):
    claim_lines = [
        f"R{review_no}-{claim_no}: {claim['claim']}"
        for review_no, review_claims in enumerate(item["claims"])
        for claim_no, claim in enumerate(review_claims)
    ]
    if claim_lines:
        panel_text = "\n".join(claim_lines)
        out = call(instructions, panel_text, PASS2_SCHEMA, "clusters", model=model, effort=effort)
    else:
        out = {"clusters": []}
    return dict(panel_id=item["panel_id"], model=model, effort=effort, review_ids=item["review_ids"], **out)


def pass1(
    reviews,
    model=settings.DEFAULT_MODEL,
    tag="",
    threads=48,
    budget=60.0,
    paper_kind="ml",
    data=str(settings.WORK_DIR),
    redo=False,
):
    effort = reasoning_efforts(model).pass1
    items = pd.read_parquet(reviews, columns=["review_id", "text"]).to_dict("records")
    worker = partial(pass1_worker, model=model, instructions=pass1_instructions(paper_kind), effort=effort)
    data = Path(data)
    done = set() if redo else done_ids(data / f"claims_{model}.jsonl", review_key, effort)
    run(items, worker, data / f"claims_{model}{tag}.jsonl", done, review_key, model, effort, threads, budget)


def panel_items(panel_rows, claims_by_review):
    items = [
        dict(
            panel_id=panel["panel_id"],
            review_ids=panel["review_ids"],
            claims=[claims_by_review[review_id] for review_id in panel["review_ids"]],
        )
        for panel in panel_rows
        if all(review_id in claims_by_review for review_id in panel["review_ids"])
    ]
    if len(items) < len(panel_rows):
        print(f"{len(panel_rows) - len(items)} panels skipped: a review has no claims (pass-1 error or not run)", flush=True)
    return items


def pass2(
    panels,
    model=settings.DEFAULT_MODEL,
    tag="",
    threads=48,
    budget=60.0,
    paper_kind="ml",
    claims_file=None,
    data=str(settings.WORK_DIR),
    redo=False,
):
    effort = reasoning_efforts(model).pass2
    data = Path(data)
    if claims_file is None:
        claims_by_review = load_claims(data / f"claims_{model}.jsonl", data / f"claims_{model}{tag}.jsonl")
    else:
        claims_by_review = load_claims(data / claims_file)
    items = panel_items(list(read_jsonl(panels)), claims_by_review)
    worker = partial(
        pass2_worker, model=model, instructions=render("pass2", paper=PASS2_PAPER_WORDING[paper_kind]), effort=effort
    )
    done = set() if redo else done_ids(data / f"clusters_{model}.jsonl", panel_key, effort)
    run(items, worker, data / f"clusters_{model}{tag}.jsonl", done, panel_key, model, effort, threads, budget)


def final_pass1(model=settings.DEFAULT_MODEL, threads=256, budget=60.0, redo=False):
    hub.check_model(model)
    effort = reasoning_efforts(model).pass1
    settings.WORK_DIR.mkdir(parents=True, exist_ok=True)
    done = set() if redo else hub.reviews_with_claims()
    for paper_kind in PASS1_PAPER_WORDING:
        items = hub.sampled_reviews(journal=paper_kind == "general", without_claims=not redo)
        worker = partial(pass1_worker, model=model, instructions=pass1_instructions(paper_kind), effort=effort)
        run(items, worker, settings.WORK_DIR / f"claims_{model}.jsonl", done, review_key, model, effort, threads, budget)


def final_pass2(model=settings.DEFAULT_MODEL, threads=160, budget=60.0, redo=False):
    hub.check_model(model)
    effort = reasoning_efforts(model).pass2
    settings.WORK_DIR.mkdir(parents=True, exist_ok=True)
    new_claims = load_claims(settings.WORK_DIR / f"claims_{model}.jsonl")
    done = set() if redo else hub.grouped_panels()
    for paper_kind in PASS2_PAPER_WORDING:
        rows = [row for row in hub.panel_rows(journal=paper_kind == "general") if panel_key(row) not in done]
        review_ids = {review_id for row in rows for review_id in row["review_ids"]}
        items = panel_items(rows, {**hub.claims(model, review_ids), **new_claims})
        worker = partial(
            pass2_worker, model=model, instructions=render("pass2", paper=PASS2_PAPER_WORDING[paper_kind]), effort=effort
        )
        run(items, worker, settings.WORK_DIR / f"clusters_{model}.jsonl", done, panel_key, model, effort, threads, budget)


if __name__ == "__main__":
    fire.Fire({"pass1": pass1, "pass2": pass2, "final_pass1": final_pass1, "final_pass2": final_pass2})
