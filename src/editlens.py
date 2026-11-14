import json
import re
from pathlib import Path

import emoji
import fire
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from ai import TAGS
from jsonl import read_jsonl
from meld import KNOWN_SETS, REAL_SETS
from settings import settings

FLUSH_DOCS = 2000
BOILERPLATE_STARTS = ("Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly")


def clean_text(text):
    text = emoji.demojize(text)
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) > 1:
        first = emoji.replace_emoji(re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0]), "")
        if first.startswith(BOILERPLATE_STARTS):
            text = "\n".join(paragraphs[1:])
    return re.sub(r"\s+", " ", text.lower()).strip()


class NormedLinear(torch.nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, x):
        return self.linear(self.norm(x))


def load_scorer():
    adapter_dir = snapshot_download(settings.EDITLENS_REPO, revision=settings.EDITLENS_REVISION)
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(
        settings.EDITLENS_BASE_REPO,
        revision=settings.EDITLENS_BASE_REVISION,
        num_labels=settings.EDITLENS_BUCKETS,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
    )
    base.config.pad_token_id = tok.pad_token_id
    base.score = NormedLinear(base.config.hidden_size, settings.EDITLENS_BUCKETS).to(base.device)
    return tok, PeftModel.from_pretrained(base, adapter_dir).eval()


@torch.no_grad()
def score_docs(tok, model, texts, batch_size):
    size = settings.EDITLENS_MAX_LENGTH - 1
    parts = []
    for doc, text in enumerate(texts):
        ids = tok(clean_text(text), add_special_tokens=False, truncation=False)["input_ids"]
        parts += [(doc, ids[i : i + size]) for i in range(0, max(len(ids), 1), size)]
    order = sorted(range(len(parts)), key=lambda i: len(parts[i][1]))
    values = torch.arange(model.config.num_labels, dtype=torch.float32) / (model.config.num_labels - 1)
    window_scores = [0.0] * len(parts)
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        width = max(len(parts[i][1]) for i in batch) + 1
        inp = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for j, i in enumerate(batch):
            ids = [tok.bos_token_id, *parts[i][1]]
            inp[j, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            att[j, width - len(ids) :] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp.to("cuda"), attention_mask=att.to("cuda")).logits
        for i, score in zip(batch, (logits.float().softmax(-1).cpu() @ values).tolist()):
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


def check(reference, n=2000, batch_size=16, column="editlens_Llama-3.2-3B_score"):
    df = pd.read_csv(reference, usecols=["text", "text_type", column]).dropna().sample(n, random_state=0)
    tok, model = load_scorer()
    df["score"] = [r["score"] for r in score_docs(tok, model, df.text.tolist(), batch_size)]
    diff = (df.score - df[column]).abs()
    print(f"{n} texts: max |ours - Pangram| {diff.max():.2e}, mean {diff.mean():.2e}, r {df.score.corr(df[column]):.5f}")
    print(df.groupby("text_type")[["score", column]].mean().round(3).to_string())


def scores(
    input=settings.RAW_DIR / "meld_input.parquet",
    out=settings.RAW_DIR / "editlens_scores.jsonl",
    batch_size=16,
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
    tok, model = load_scorer()
    with open(out, "a") as f:
        for start in range(0, len(df), FLUSH_DOCS):
            chunk = df.iloc[start : start + FLUSH_DOCS]
            for key, result in zip(chunk.key, score_docs(tok, model, chunk.text.tolist(), batch_size)):
                f.write(json.dumps(dict(key=key, **result)) + "\n")
            f.flush()
            print(f"{start + len(chunk)}/{len(df)}", flush=True)


REVIEW_ROWS = [
    ("ICLR", "sample", "ICLR", range(2024, 2028)),
    ("NeurIPS", "sample", "NeurIPS", range(2023, 2026)),
    ("ICML", "icml", "ICML", range(2025, 2027)),
    ("Journals", "journal", None, range(2023, 2027)),
]


def compare_table(reports, tex, known_tex):
    known = pd.read_csv(Path(reports) / "meld_labels.csv").set_index("known")
    known = known.assign(
        meld=100 - known.unflagged, editlens=pd.read_csv(Path(reports) / "editlens_labels.csv").set_index("known").flagged
    )
    head = r"Texts & $n$ & \shortstack[r]{MELD\\flagged (\%)} & \shortstack[r]{MELD\\written (\%)} & \shortstack[r]{EditLens\\flagged (\%)} \\"
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", head, r"\midrule"]
    lines += [
        f"{KNOWN_SETS[k]} & {int(r.texts):,} & {r.meld:.1f} & {r.written:.1f} & {r.editlens:.1f} \\\\"
        for k, r in known.iterrows()
    ]
    Path(known_tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Reviews & $n$ & MELD (\%) & EditLens (\%) \\", r"\midrule"]
    rates = pd.read_csv(Path(reports) / "meld_rates.csv").merge(
        pd.read_csv(Path(reports) / "editlens_rates.csv"), on=["set", "conf", "year"], suffixes=("_meld", "_editlens")
    )
    for label, name, conf, years in REVIEW_ROWS:
        cell = rates[(rates.set == name) & (rates.conf == conf if conf else True)]
        for year in years:
            part = cell[cell.year == year]
            n = part.n_editlens.sum()
            meld, editlens = ((part[f"flagged_{d}"] * part[f"n_{d}"]).sum() / part[f"n_{d}"].sum() for d in ["meld", "editlens"])
            lines.append(f"{label} {year} & {int(n):,} & {meld:.1f} & {editlens:.1f} \\\\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def flags(
    input=settings.RAW_DIR / "meld_input.parquet",
    scored=settings.RAW_DIR / "editlens_scores.jsonl",
    out=settings.RAW_DIR / "editlens_flags.parquet",
    reports=settings.REPORTS_DIR,
    tex=settings.TABLES_DIR / "detectors_compare.tex",
    known_tex=settings.TABLES_DIR / "known_texts.tex",
):
    df = pd.read_parquet(input, columns=["key", "set", "review_id", "conf", "year"]).merge(
        pd.read_json(scored, lines=True)[["key", "score", "n_tokens"]], on="key"
    )
    is_real = df.set.isin(REAL_SETS)
    pre = is_real & (df.year <= settings.MELD_CALIBRATION_LAST_YEAR)
    threshold = df[pre].score.quantile(settings.MELD_FLAG_QUANTILE)
    df["flag"] = df.score > threshold
    rates = df.groupby(["set", "conf", "year"]).flag.agg(n="size", flagged="mean")
    rates["flagged"] *= 100
    rates.round(3).to_csv(Path(reports) / "editlens_rates.csv")
    print(rates.round(2).to_string())
    heldout = is_real & (df.conf == "ICLR") & (df.year == settings.MELD_CALIBRATION_LAST_YEAR + 1)
    known = df.set.where(~is_real).mask(pre, "human").mask(heldout, "heldout")
    labels = df.groupby(known).flag.agg(texts="size", flagged="mean").rename_axis("known")
    labels["flagged"] *= 100
    labels = labels.loc[[k for k in KNOWN_SETS if k in labels.index]]
    labels.round(3).to_csv(Path(reports) / "editlens_labels.csv")
    print(labels.round(1).to_string())
    real = df[is_real]
    assert real.review_id.is_unique
    meld = pd.read_parquet(settings.RAW_DIR / "meld_flags.parquet", columns=["review_id", "score", "flag"])
    meld = meld.set_index("review_id").rename(columns={"score": "meld_score", "flag": "meld"})
    post = real[real.year > settings.MELD_CALIBRATION_LAST_YEAR].join(meld, on="review_id")
    agree = (post.flag == post.meld).mean()
    chance = post.flag.mean() * post.meld.mean() + (1 - post.flag.mean()) * (1 - post.meld.mean())
    summary = {
        "threshold": threshold,
        "calibration_reviews": int(pre.sum()),
        "pre_source_max": df[pre].groupby("conf").flag.mean().max() * 100,
        "agreement_reviews": len(post),
        "both": (post.flag & post.meld).mean() * 100,
        "meld_only": (~post.flag & post.meld).mean() * 100,
        "editlens_only": (post.flag & ~post.meld).mean() * 100,
        "kappa": (agree - chance) / (1 - chance),
        "spearman": post.score.corr(post.meld_score, method="spearman"),
    }
    pd.Series(summary).to_csv(Path(reports) / "editlens_summary.csv", header=["value"], index_label="name")
    print(pd.Series(summary).round(3).to_string())
    real[["review_id", "conf", "year", "score", "n_tokens", "flag"]].to_parquet(out, index=False)
    compare_table(reports, tex, known_tex)


if __name__ == "__main__":
    fire.Fire({"check": check, "scores": scores, "flags": flags})
