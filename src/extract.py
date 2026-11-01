import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import fire
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import render
from settings import settings

PAPER_KIND = {
    "ml": ("machine-learning conference paper", "machine-learning paper"),
    "general": ("scientific paper submitted to a journal", "scientific paper"),
}

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


def pass1_instructions(paper_kind="ml"):
    return render("pass1", paper=PAPER_KIND[paper_kind][0])


PASS1_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "category", "target", "specific"],
                "properties": {
                    "claim": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "target": {"type": "string", "enum": TARGETS},
                    "specific": {"type": "boolean"},
                },
            },
        }
    },
}


def pass2_instructions(paper_kind="ml", different=False):
    return render("pass2", paper=PAPER_KIND[paper_kind][1], different=different)


PASS2_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["targets", "clusters"],
    "properties": {
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target_id", "element"],
                "properties": {"target_id": {"type": "integer"}, "element": {"type": "string"}},
            },
        },
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target_id", "problem", "claim_ids"],
                "properties": {
                    "target_id": {"type": "integer"},
                    "problem": {"type": "string"},
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

_client = None
_openrouter = None
_llm = None
spent = {}
lock = threading.Lock()


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=6, timeout=180)
    return _client


def openrouter_client():
    global _openrouter
    if _openrouter is None:
        from openai import OpenAI

        _openrouter = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=settings.OPENROUTER_API_KEY, max_retries=6, timeout=180
        )
    return _openrouter


def openrouter_call(instructions, text, schema, name, model, effort):
    spec = settings.OPENROUTER_ROUTES[model]
    provider = spec.provider or {"require_parameters": True}
    if spec.strict:
        fmt = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
    else:
        fmt = {"type": "json_object"}
        instructions += f"\n\nReturn one JSON object that follows this JSON schema exactly:\n{json.dumps(schema)}"
    for temperature in (None,) if spec.strict else (0.0, 0.8, 0.8, 0.8):
        r = openrouter_client().chat.completions.create(
            model=spec.id,
            **({} if temperature is None else {"temperature": temperature}),
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": text}],
            response_format=fmt,
            extra_body={"reasoning": {"effort": effort}, "provider": provider},
        )
        record(model, r.usage.prompt_tokens, r.usage.completion_tokens)
        try:
            return parse_object(r.choices[0].message.content, schema)
        except ValueError as e:
            err = e
    raise err


def parse_object(text, schema=None):
    out = json.loads(text)
    if not isinstance(out, dict):
        raise ValueError(f"response is not a JSON object: {text[:80]}")
    if schema is not None and not conforms(out, schema):
        raise ValueError(f"response does not follow the schema: {text[:80]}")
    return out


def conforms(x, s):
    if "enum" in s:
        return x in s["enum"]
    t = s["type"]
    if t == "object":
        return (
            isinstance(x, dict)
            and all(k in x for k in s.get("required", []))
            and all(k in s["properties"] and conforms(v, s["properties"][k]) for k, v in x.items())
        )
    if t == "array":
        return isinstance(x, list) and all(conforms(v, s["items"]) for v in x)
    return isinstance(x, {"string": str, "integer": int, "boolean": bool}[t]) and not (t == "integer" and isinstance(x, bool))


def record(model, n_in, n_out):
    with lock:
        s = spent.setdefault(model, {"in": 0, "out": 0, "calls": 0})
        s["in"] += n_in
        s["out"] += n_out
        s["calls"] += 1


def cost():
    total = 0.0
    with lock:
        for m, s in spent.items():
            price_in, price_out = settings.PRICES_USD_PER_MTOK.get(m, (0.0, 0.0))
            total += (s["in"] * price_in + s["out"] * price_out) / 1e6
    return total


def local_call(instructions, text, schema, max_tokens, model, local_model_path):
    global _llm
    from llama_cpp import Llama

    with lock:
        if _llm is None:
            _llm = Llama(model_path=local_model_path, n_ctx=8192, n_threads=os.cpu_count(), verbose=False)
        r = _llm.create_chat_completion(
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": text}],
            response_format={"type": "json_object", "schema": schema},
            temperature=0.0,
            max_tokens=max_tokens,
        )
    record(model, r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"])
    return json.loads(r["choices"][0]["message"]["content"])


def call(
    instructions, text, schema, name, model=settings.DEFAULT_MODEL, effort=settings.PASS1_REASONING_EFFORT, local_model_path=None
):
    if local_model_path:
        try:
            return local_call(instructions, text, schema, 1200 if name == "claims" else 2500, model, local_model_path)
        except Exception as e:
            return {"error": repr(e)[:300]}
    for attempt in range(4):
        try:
            if model in settings.OPENROUTER_ROUTES:
                return openrouter_call(instructions, text, schema, name, model, effort)
            resp = client().responses.create(
                model=model,
                instructions=instructions,
                input=text,
                reasoning={"effort": effort},
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            )
            record(model, resp.usage.input_tokens, resp.usage.output_tokens)
            return parse_object(resp.output_text)
        except Exception as e:
            if attempt == 3:
                return {"error": repr(e)[:300]}
            time.sleep(5 * (attempt + 1))


def done_ids(path, key):
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in open(path) if line.strip())
    return {r[key] for r in rows if "error" not in r}


def run(items, worker, out_path, key, model, threads, budget, local_model_path):
    if not local_model_path and model not in settings.PRICES_USD_PER_MTOK:
        raise SystemExit(f"{model} has no entry in settings.PRICES_USD_PER_MTOK, so the budget cap would not count its calls")
    skip = done_ids(out_path, key)
    todo = [it for it in items if it[key] not in skip]
    print(f"{model}: {len(skip)} done, {len(todo)} to do -> {out_path}", flush=True)

    def capped(item):
        return None if cost() > budget else worker(item)

    pool = ThreadPoolExecutor(threads)
    try:
        with open(out_path, "a") as f:
            bar = tqdm(as_completed([pool.submit(capped, it) for it in todo]), total=len(todo), desc=out_path.stem)
            for fut in bar:
                res = fut.result()
                if res is not None:
                    f.write(json.dumps(res) + "\n")
                    f.flush()
                bar.set_postfix(cost=f"${cost():.2f}")
    finally:
        pool.shutdown(cancel_futures=True)
    if cost() > budget:
        print("budget cap reached", flush=True)
    s = spent.get(model, {"in": 0, "out": 0, "calls": 0})
    print(f"finished: {s['calls']} calls, {s['in'] / 1e6:.2f}M in, {s['out'] / 1e6:.2f}M out, ${cost():.2f}", flush=True)


def pass1_worker(item, model, instructions, local_model_path, effort):
    out = call(
        instructions,
        item["text"][: settings.REVIEW_TEXT_MAX_CHARS],
        PASS1_SCHEMA,
        "claims",
        model=model,
        effort=effort,
        local_model_path=local_model_path,
    )
    return dict(review_id=item["review_id"], model=model, **out)


def pass2_worker(item, model, instructions, local_model_path, effort):
    lines = [
        f"R{j}-{k} [{c['category']}; {c['target']}]: {c['claim']}"
        for j, claims in enumerate(item["claims"])
        for k, c in enumerate(claims)
    ]
    if not lines:
        return dict(panel_id=item["panel_id"], model=model, review_ids=item["review_ids"], clusters=[])
    out = call(
        instructions, "\n".join(lines), PASS2_SCHEMA, "clusters", model=model, effort=effort, local_model_path=local_model_path
    )
    return dict(panel_id=item["panel_id"], model=model, review_ids=item["review_ids"], **out)


def load_claims(path):
    claims = {}
    for line in open(path):
        r = json.loads(line)
        if "claims" in r:
            claims[r["review_id"]] = r["claims"]
    return claims


def pass1(
    reviews,
    model=settings.DEFAULT_MODEL,
    tag="",
    threads=48,
    budget=60.0,
    paper_kind="ml",
    local_model_path=None,
    data=str(settings.DATA_DIR),
    effort=settings.PASS1_REASONING_EFFORT,
):
    import pandas as pd

    items = pd.read_parquet(reviews, columns=["review_id", "text"]).to_dict("records")
    worker = partial(
        pass1_worker, model=model, instructions=pass1_instructions(paper_kind), local_model_path=local_model_path, effort=effort
    )
    run(items, worker, Path(data) / f"claims_{model}{tag}.jsonl", "review_id", model, threads, budget, local_model_path)


def pass2(
    panels,
    model=settings.DEFAULT_MODEL,
    tag="",
    threads=48,
    budget=60.0,
    paper_kind="ml",
    local_model_path=None,
    claims_files=None,
    effort=settings.PASS2_REASONING_EFFORT,
    data=str(settings.DATA_DIR),
    different=False,
):
    if claims_files is None:
        claims_files = f"claims_{model}{tag}.jsonl"
    if isinstance(claims_files, str):
        claims_files = claims_files.split(",")
    claims = {}
    for name in claims_files:
        claims.update(load_claims(Path(data) / name))
    panels = [json.loads(line) for line in open(panels)]
    items = [
        dict(panel_id=p["panel_id"], review_ids=p["review_ids"], claims=[claims[r] for r in p["review_ids"]])
        for p in panels
        if all(r in claims for r in p["review_ids"])
    ]
    if len(items) < len(panels):
        print(f"{len(panels) - len(items)} panels skipped: a review has no claims (pass-1 error or not run)", flush=True)
    worker = partial(
        pass2_worker,
        model=model,
        instructions=pass2_instructions(paper_kind, different),
        local_model_path=local_model_path,
        effort=effort,
    )
    run(items, worker, Path(data) / f"clusters_{model}{tag}.jsonl", "panel_id", model, threads, budget, local_model_path)


if __name__ == "__main__":
    fire.Fire({"pass1": pass1, "pass2": pass2})
