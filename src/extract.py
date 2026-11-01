import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache, partial
from pathlib import Path

import fire
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import render
from settings import settings

PASS1_PAPER_WORDING = {"ml": "machine-learning conference paper", "general": "scientific paper submitted to a journal"}
PASS2_PAPER_WORDING = {"ml": "machine-learning paper", "general": "scientific paper"}

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


def pass1_instructions(paper_kind="ml"):
    return render("pass1", paper=PASS1_PAPER_WORDING[paper_kind])


def pass2_instructions(paper_kind="ml", different=False):
    return render("pass2", paper=PASS2_PAPER_WORDING[paper_kind], different=different)


JSON_TYPES = {"string": str, "integer": int, "boolean": bool}

CLIENT_MAX_RETRIES = 6
CLIENT_TIMEOUT_SECONDS = 180
API_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 5
ERROR_REPR_CHARS = 300
NON_STRICT_TEMPERATURES = (0.0, 0.8, 0.8, 0.8)
LOCAL_CONTEXT_TOKENS = 8192
LOCAL_CLAIMS_MAX_TOKENS = 1200
LOCAL_DEFAULT_MAX_TOKENS = 2500

_llm = None
llama_lock = threading.Lock()
usage_by_model = {}
usage_lock = threading.Lock()


@cache
def openai_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=CLIENT_MAX_RETRIES, timeout=CLIENT_TIMEOUT_SECONDS)


@cache
def openrouter_client():
    from openai import OpenAI

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


def openrouter_call(instructions, text, schema, name, model, effort):
    route = settings.OPENROUTER_ROUTES[model]
    provider = route.provider or {"require_parameters": True}
    if route.strict:
        response_format = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        sampling_options = [{}]
    else:
        response_format = {"type": "json_object"}
        instructions += f"\n\nReturn one JSON object that follows this JSON schema exactly:\n{json.dumps(schema)}"
        sampling_options = [{"temperature": temperature} for temperature in NON_STRICT_TEMPERATURES]
    for sampling in sampling_options:
        response = openrouter_client().chat.completions.create(
            model=route.id,
            **sampling,
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": text}],
            response_format=response_format,
            extra_body={"reasoning": {"effort": effort}, "provider": provider},
        )
        record(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        try:
            return parse_object(response.choices[0].message.content, schema)
        except ValueError as e:
            last_error = e
    raise last_error


def local_call(instructions, text, schema, max_tokens, model, local_model_path):
    global _llm
    from llama_cpp import Llama

    with llama_lock:
        if _llm is None:
            _llm = Llama(model_path=local_model_path, n_ctx=LOCAL_CONTEXT_TOKENS, n_threads=os.cpu_count(), verbose=False)
        response = _llm.create_chat_completion(
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": text}],
            response_format={"type": "json_object", "schema": schema},
            temperature=0.0,
            max_tokens=max_tokens,
        )
    record(model, response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"])
    return json.loads(response["choices"][0]["message"]["content"])


def error_row(e):
    return {"error": repr(e)[:ERROR_REPR_CHARS]}


def call(
    instructions, text, schema, name, model=settings.DEFAULT_MODEL, effort=settings.PASS1_REASONING_EFFORT, local_model_path=None
):
    if local_model_path:
        try:
            max_tokens = LOCAL_CLAIMS_MAX_TOKENS if name == "claims" else LOCAL_DEFAULT_MAX_TOKENS
            return local_call(instructions, text, schema, max_tokens, model, local_model_path)
        except Exception as e:
            return error_row(e)
    for attempt in range(API_ATTEMPTS):
        try:
            if model in settings.OPENROUTER_ROUTES:
                return openrouter_call(instructions, text, schema, name, model, effort)
            return openai_call(instructions, text, schema, name, model, effort)
        except Exception as e:
            if attempt == API_ATTEMPTS - 1:
                return error_row(e)
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def load_claims(path):
    return {row["review_id"]: row["claims"] for row in read_jsonl(path) if "claims" in row}


def done_ids(path, key):
    if not path.exists():
        return set()
    with open(path) as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {row[key] for row in rows if "error" not in row}


def run(items, worker, out_path, key, model, threads, budget, local_model_path):
    if not local_model_path and model not in settings.PRICES_USD_PER_MTOK:
        raise SystemExit(f"{model} has no entry in settings.PRICES_USD_PER_MTOK, so the budget cap would not count its calls")
    done = done_ids(out_path, key)
    todo = [item for item in items if item[key] not in done]
    print(f"{model}: {len(done)} done, {len(todo)} to do -> {out_path}", flush=True)

    def capped(item):
        return None if cost() > budget else worker(item)

    pool = ThreadPoolExecutor(threads)
    try:
        with open(out_path, "a") as f:
            bar = tqdm(as_completed([pool.submit(capped, item) for item in todo]), total=len(todo), desc=out_path.stem)
            for future in bar:
                row = future.result()
                if row is not None:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                bar.set_postfix(cost=f"${cost():.2f}")
    finally:
        pool.shutdown(cancel_futures=True)
    if cost() > budget:
        print("budget cap reached", flush=True)
    usage = usage_by_model.get(model, empty_usage())
    print(
        f"finished: {usage['calls']} calls, {usage['in'] / 1e6:.2f}M in, {usage['out'] / 1e6:.2f}M out, ${cost():.2f}", flush=True
    )


def pass1_worker(item, model, instructions, local_model_path, effort):
    review_text = item["text"][: settings.REVIEW_TEXT_MAX_CHARS]
    out = call(instructions, review_text, PASS1_SCHEMA, "claims", model=model, effort=effort, local_model_path=local_model_path)
    return dict(review_id=item["review_id"], model=model, **out)


def pass2_worker(item, model, instructions, local_model_path, effort):
    claim_lines = [
        f"R{review_no}-{claim_no} [{claim['category']}; {claim['target']}]: {claim['claim']}"
        for review_no, review_claims in enumerate(item["claims"])
        for claim_no, claim in enumerate(review_claims)
    ]
    if claim_lines:
        panel_text = "\n".join(claim_lines)
        out = call(
            instructions, panel_text, PASS2_SCHEMA, "clusters", model=model, effort=effort, local_model_path=local_model_path
        )
    else:
        out = {"clusters": []}
    return dict(panel_id=item["panel_id"], model=model, review_ids=item["review_ids"], **out)


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
    claims_by_review = {}
    for claims_file in claims_files:
        claims_by_review.update(load_claims(Path(data) / claims_file))
    panel_rows = list(read_jsonl(panels))
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
