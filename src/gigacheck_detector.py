import json
from pathlib import Path

import fire
import pandas as pd
import torch
from transformers import AutoModel

from ai import TAGS
from gpttrace import flag_reports
from jsonl import read_jsonl
from settings import settings

FLUSH_DOCS = 5000


def load_scorer():
    model = AutoModel.from_pretrained(
        settings.GIGACHECK_REPO,
        revision=settings.GIGACHECK_REVISION,
        trust_remote_code=True,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model


@torch.no_grad()
def score_docs(model, texts, batch_size):
    backbone = type(model).__mro__[1]
    ai_label = next(int(k) for k, v in model.config.id2label.items() if v == "ai")
    lengths = [len(ids) for ids in model.tokenizer(texts, add_special_tokens=False).input_ids]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])
    results = [dict(score=None, n_tokens=n) for n in lengths]
    for start in range(0, len(order), batch_size):
        batch = [i for i in order[start : start + batch_size] if lengths[i]]
        if not batch:
            continue
        ids, mask, _ = model._get_inputs([texts[i] for i in batch])
        logits = backbone.forward(model, input_ids=ids, attention_mask=mask, return_dict=True, return_detr_output=False).logits
        logits = logits.float().cpu()
        for i, score in zip(batch, (logits[:, ai_label] - logits[:, 1 - ai_label]).tolist()):
            results[i]["score"] = score
    return results


def scores(
    input=settings.RAW_DIR / "meld_input.parquet",
    out=settings.RAW_DIR / "gigacheck_scores.jsonl",
    batch_size=32,
    per_set=None,
):
    out = Path(out)
    df = pd.read_parquet(input, columns=["key", "set", "text"])
    if per_set:
        df = df.groupby("set").sample(per_set, random_state=0)
    done = {row["key"] for row in read_jsonl(out)} if out.exists() else set()
    df = df[~df.key.isin(done)]
    df = df.assign(text=df.text.str.replace(TAGS, " ", regex=True), n=df.text.str.len()).sort_values("n")
    print(f"{len(done)} done, {len(df)} to score -> {out}", flush=True)
    model = load_scorer()
    with open(out, "a") as f:
        for start in range(0, len(df), FLUSH_DOCS):
            chunk = df.iloc[start : start + FLUSH_DOCS]
            for key, result in zip(chunk.key, score_docs(model, chunk.text.tolist(), batch_size)):
                f.write(json.dumps(dict(key=key, **result)) + "\n")
            f.flush()
            print(f"{start + len(chunk)}/{len(df)}", flush=True)


def flags(
    input=settings.RAW_DIR / "meld_input.parquet",
    scored=settings.RAW_DIR / "gigacheck_scores.jsonl",
    out=settings.RAW_DIR / "gigacheck_flags.parquet",
    reports=settings.REPORTS_DIR,
):
    flag_reports("gigacheck", ["score", "n_tokens"], ["meld", "editlens", "gpttrace"], input, scored, out, reports)


if __name__ == "__main__":
    fire.Fire({"scores": scores, "flags": flags})
