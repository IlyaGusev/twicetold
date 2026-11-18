from functools import partial
from pathlib import Path

import fire
import pandas as pd
import torch
from transformers import AutoModel

from meld import KNOWN_SETS, flag_reports, score_file
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
    score_file(input, out, partial(score_docs, load_scorer(), batch_size=batch_size), FLUSH_DOCS, per_set)


REVIEW_ROWS = [
    ("ICLR", "sample", "ICLR", range(2024, 2028)),
    ("NeurIPS", "sample", "NeurIPS", range(2023, 2026)),
    ("ICML", "icml", "ICML", range(2025, 2027)),
    ("Journals", "journal", None, range(2023, 2027)),
]


KNOWN_GROUP_STARTS = {"polish", "rewrite_panel", "intel_ai"}


def compare_table(
    reports=settings.REPORTS_DIR,
    tex=settings.TABLES_DIR / "detectors_compare.tex",
    known_tex=settings.TABLES_DIR / "known_texts.tex",
):
    known = pd.read_csv(Path(reports) / "meld_labels.csv").set_index("known")
    known = known.assign(
        meld=100 - known.unflagged, gigacheck=pd.read_csv(Path(reports) / "gigacheck_labels.csv").set_index("known").flagged
    )
    head = r"Texts & $n$ & \shortstack[r]{MELD\\flagged (\%)} & \shortstack[r]{GigaCheck\\flagged (\%)} \\"
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", head, r"\midrule"]
    for k, r in known.iterrows():
        lines += [r"\midrule"] * (k in KNOWN_GROUP_STARTS)
        lines.append(f"{KNOWN_SETS[k]} & {int(r.texts):,} & {r.meld:.1f} & {r.gigacheck:.1f} \\\\")
    Path(known_tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Reviews & $n$ & MELD (\%) & GigaCheck (\%) \\", r"\midrule"]
    rates = pd.read_csv(Path(reports) / "meld_rates.csv").merge(
        pd.read_csv(Path(reports) / "gigacheck_rates.csv"), on=["set", "conf", "year"], suffixes=("_meld", "_gigacheck")
    )
    for k, (label, name, conf, years) in enumerate(REVIEW_ROWS):
        lines += [r"\midrule"] * (k > 0)
        cell = rates[(rates.set == name) & (rates.conf == conf if conf else True)]
        for year in years:
            part = cell[cell.year == year]
            n = part.n_gigacheck.sum()
            meld, gigacheck = (
                (part[f"flagged_{d}"] * part[f"n_{d}"]).sum() / part[f"n_{d}"].sum() for d in ["meld", "gigacheck"]
            )
            lines.append(f"{label} {year} & {int(n):,} & {meld:.1f} & {gigacheck:.1f} \\\\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def flags(
    input=settings.RAW_DIR / "meld_input.parquet",
    scored=settings.RAW_DIR / "gigacheck_scores.jsonl",
    out=settings.RAW_DIR / "gigacheck_flags.parquet",
    reports=settings.REPORTS_DIR,
    tex=settings.TABLES_DIR / "detectors_compare.tex",
    known_tex=settings.TABLES_DIR / "known_texts.tex",
):
    flag_reports("gigacheck", ["score", "n_tokens"], ["meld", "editlens", "gpttrace"], input, scored, out, reports)
    compare_table(reports, tex, known_tex)


if __name__ == "__main__":
    fire.Fire({"scores": scores, "flags": flags, "compare_table": compare_table})
