import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import pandas as pd

from settings import settings

CLASSES = ["human", "mix", "ai"]
TAGS = r"\[(SUMMARY|MAIN|STRENGTHS|WEAKNESSES|QUESTIONS|LIMITATIONS|OTHER)\]\s*"
LOGIT_SCALE = 30.0
SORT_CHUNK = 256
FLUSH_BATCHES = 250


def load_model():
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from torch import nn
    from transformers import AutoModel

    class CoCoDet(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(settings.COCODET_BASE_MODEL)
            hidden = self.bert.config.hidden_size
            self.main_classifier_weight = nn.Parameter(torch.empty(len(CLASSES), hidden))
            self.content_classifier = nn.Linear(hidden, 8)
            self.style_classifier = nn.Linear(hidden, 7)
            self.fine_grained_classifier = nn.Linear(hidden, 6)

        def forward(self, input_ids, attention_mask):
            pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            cos = F.linear(F.normalize(pooled, dim=1), F.normalize(self.main_classifier_weight, dim=1))
            return (cos * LOGIT_SCALE).softmax(-1)

    model = CoCoDet()
    weights = hf_hub_download(settings.COCODET_REPO, settings.COCODET_WEIGHTS_FILE)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


def scores(input, out, max_len=512, batch=8, limit=None):
    import torch
    from transformers import AutoTokenizer

    out = Path(out)
    df = pd.read_parquet(input, columns=["review_id", "text"])
    done = {json.loads(line)["review_id"] for line in open(out)} if out.exists() else set()
    df = df[~df.review_id.isin(done)]
    if limit is not None:
        df = df.head(limit)
    df = df.assign(text=df.text.str.replace(TAGS, "", regex=True)).sample(frac=1, random_state=0)
    df["chunk"] = range(len(df))
    df["chunk"] //= SORT_CHUNK
    df = df.assign(n=df.text.str.len()).sort_values(["chunk", "n"])
    print(f"{len(done)} done, {len(df)} to score -> {out}", flush=True)

    tok = AutoTokenizer.from_pretrained(settings.COCODET_BASE_MODEL)
    model = load_model()
    start_time = time.time()
    with open(out, "a") as f, torch.inference_mode():
        for i in range(0, len(df), batch):
            rows = df.iloc[i : i + batch]
            enc = tok(rows.text.tolist(), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            probs = model(enc["input_ids"], enc["attention_mask"]).tolist()
            for review_id, p in zip(rows.review_id, probs):
                f.write(json.dumps(dict(review_id=review_id, **dict(zip(CLASSES, p)))) + "\n")
            if (i // batch) % FLUSH_BATCHES == 0:
                f.flush()
                rate = (i + len(rows)) / (time.time() - start_time)
                print(f"  {i + len(rows)}/{len(df)} {rate:.1f}/s", flush=True)
    print(f"finished in {(time.time() - start_time) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    fire.Fire({"scores": scores})
