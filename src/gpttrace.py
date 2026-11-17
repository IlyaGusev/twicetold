import importlib.util
from functools import partial
from pathlib import Path

import fire
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from meld import flag_reports, merge_windows, score_file
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
    return merge_windows(parts, window_scores, len(texts))


def scores(
    input=settings.RAW_DIR / "meld_input.parquet",
    out=settings.RAW_DIR / "gpttrace_scores.jsonl",
    batch_size=64,
    per_set=None,
):
    normalize, tok, model = load_scorer()
    score_file(input, out, partial(score_docs, normalize, tok, model, batch_size=batch_size), FLUSH_DOCS, per_set)


def flags(
    input=settings.RAW_DIR / "meld_input.parquet",
    scored=settings.RAW_DIR / "gpttrace_scores.jsonl",
    out=settings.RAW_DIR / "gpttrace_flags.parquet",
    reports=settings.REPORTS_DIR,
):
    flag_reports("gpttrace", ["score", "score_mean", "n_tokens"], ["meld", "editlens"], input, scored, out, reports)


if __name__ == "__main__":
    fire.Fire({"scores": scores, "flags": flags})
