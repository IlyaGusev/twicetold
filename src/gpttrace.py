import importlib.util
import json
from pathlib import Path

import fire
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ai import TAGS
from jsonl import read_jsonl
from meld import KNOWN_SETS, REAL_SETS
from settings import settings

FLUSH_DOCS = 5000


def load_scorer():
    model_dir = Path(
        snapshot_download(settings.GPTTRACE_REPO, revision=settings.GPTTRACE_REVISION, allow_patterns=["model/*", "normalize.py"])
    )
    spec = importlib.util.spec_from_file_location("gpttrace_normalize", model_dir / "normalize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tok = AutoTokenizer.from_pretrained(model_dir / "model")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir / "model", dtype=torch.bfloat16)
    return module.normalize_text, tok, model.cuda().eval()


@torch.no_grad()
def score_docs(normalize, tok, model, texts, batch_size):
    size = settings.GPTTRACE_MAX_LENGTH - 2
    parts = []
    for doc, text in enumerate(texts):
        ids = tok(normalize(text), add_special_tokens=False, truncation=False)["input_ids"]
        parts += [(doc, ids[i : i + size]) for i in range(0, max(len(ids), 1), size)]
    order = sorted(range(len(parts)), key=lambda i: len(parts[i][1]))
    window_scores = [0.0] * len(parts)
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        width = max(len(parts[i][1]) for i in batch) + 2
        inp = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for j, i in enumerate(batch):
            ids = [tok.cls_token_id, *parts[i][1], tok.sep_token_id]
            inp[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            att[j, : len(ids)] = 1
        logits = model(input_ids=inp.to("cuda"), attention_mask=att.to("cuda")).logits.float().cpu()
        for i, score in zip(batch, (logits[:, 1] - logits[:, 0]).tolist()):
            window_scores[i] = score
    results = [dict(score=None, score_mean=None, n_tokens=0, n_windows=0) for _ in texts]
    for (doc, ids), score in zip(parts, window_scores):
        r = results[doc]
        if r["score"] is None:
            r.update(score=score, score_mean=0.0)
        r["score_mean"] += score * len(ids)
        r["n_tokens"] += len(ids)
        r["n_windows"] += 1
    for r in results:
        r["score_mean"] = r["score_mean"] / r["n_tokens"] if r["n_tokens"] else None
    return results


def scores(
    input=settings.RAW_DIR / "meld_input.parquet",
    out=settings.RAW_DIR / "gpttrace_scores.jsonl",
    batch_size=64,
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
    normalize, tok, model = load_scorer()
    with open(out, "a") as f:
        for start in range(0, len(df), FLUSH_DOCS):
            chunk = df.iloc[start : start + FLUSH_DOCS]
            for key, result in zip(chunk.key, score_docs(normalize, tok, model, chunk.text.tolist(), batch_size)):
                f.write(json.dumps(dict(key=key, **result)) + "\n")
            f.flush()
            print(f"{start + len(chunk)}/{len(df)}", flush=True)


def agreement(name, post, other):
    agree = (post.flag == post[other]).mean()
    chance = post.flag.mean() * post[other].mean() + (1 - post.flag.mean()) * (1 - post[other].mean())
    return {
        f"{other}_reviews": len(post),
        f"{other}_both": (post.flag & post[other]).mean() * 100,
        f"{other}_only": (~post.flag & post[other]).mean() * 100,
        f"{name}_not_{other}": (post.flag & ~post[other]).mean() * 100,
        f"{other}_kappa": (agree - chance) / (1 - chance),
        f"{other}_spearman": post.score.corr(post[f"{other}_score"], method="spearman"),
    }


def flag_reports(name, score_columns, others, input, scored, out, reports):
    df = pd.read_parquet(input, columns=["key", "set", "review_id", "conf", "year"]).merge(
        pd.read_json(scored, lines=True)[["key", *score_columns]], on="key"
    )
    is_real = df.set.isin(REAL_SETS)
    pre = is_real & (df.year <= settings.MELD_CALIBRATION_LAST_YEAR)
    threshold = df[pre].score.quantile(settings.MELD_FLAG_QUANTILE)
    df["flag"] = df.score > threshold
    rates = df.groupby(["set", "conf", "year"]).flag.agg(n="size", flagged="mean")
    rates["flagged"] *= 100
    rates.round(3).to_csv(Path(reports) / f"{name}_rates.csv")
    print(rates.round(2).to_string())
    heldout = is_real & (df.conf == "ICLR") & (df.year == settings.MELD_CALIBRATION_LAST_YEAR + 1)
    known = df.set.where(~is_real).mask(pre, "human").mask(heldout, "heldout")
    labels = df.groupby(known).flag.agg(texts="size", flagged="mean").rename_axis("known")
    labels["flagged"] *= 100
    labels = labels.loc[[k for k in KNOWN_SETS if k in labels.index]]
    labels.round(3).to_csv(Path(reports) / f"{name}_labels.csv")
    print(labels.round(1).to_string())
    real = df[is_real]
    assert real.review_id.is_unique
    post = real[real.year > settings.MELD_CALIBRATION_LAST_YEAR]
    summary = {
        "threshold": threshold,
        "calibration_reviews": int(pre.sum()),
        "pre_source_max": df[pre].groupby("conf").flag.mean().max() * 100,
    }
    for other in others:
        marks = pd.read_parquet(settings.RAW_DIR / f"{other}_flags.parquet", columns=["review_id", "score", "flag"])
        marks = marks.set_index("review_id").rename(columns={"score": f"{other}_score", "flag": other})
        summary.update(agreement(name, post.join(marks, on="review_id", how="inner"), other))
    pd.Series(summary).to_csv(Path(reports) / f"{name}_summary.csv", header=["value"], index_label="name")
    print(pd.Series(summary).round(3).to_string())
    real[["review_id", "conf", "year", *score_columns, "flag"]].to_parquet(out, index=False)


def flags(
    input=settings.RAW_DIR / "meld_input.parquet",
    scored=settings.RAW_DIR / "gpttrace_scores.jsonl",
    out=settings.RAW_DIR / "gpttrace_flags.parquet",
    reports=settings.REPORTS_DIR,
):
    flag_reports("gpttrace", ["score", "score_mean", "n_tokens"], ["meld", "editlens"], input, scored, out, reports)


if __name__ == "__main__":
    fire.Fire({"scores": scores, "flags": flags})
