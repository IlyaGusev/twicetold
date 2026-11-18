import re
from functools import partial
from pathlib import Path

import emoji
import fire
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from meld import KNOWN_SETS, flag_reports, merge_windows, score_file
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
    return merge_windows(parts, window_scores, len(texts))


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
    tok, model = load_scorer()
    score_file(input, out, partial(score_docs, tok, model, batch_size=batch_size), FLUSH_DOCS, per_set)


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
    head = r"Texts & $n$ & \shortstack[r]{MELD\\flagged (\%)} & \shortstack[r]{EditLens\\flagged (\%)} \\"
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", head, r"\midrule"]
    lines += [f"{KNOWN_SETS[k]} & {int(r.texts):,} & {r.meld:.1f} & {r.editlens:.1f} \\\\" for k, r in known.iterrows()]
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
    flag_reports("editlens", ["score", "n_tokens"], ["meld"], input, scored, out, reports)
    compare_table(reports, tex, known_tex)


if __name__ == "__main__":
    fire.Fire({"check": check, "scores": scores, "flags": flags})
