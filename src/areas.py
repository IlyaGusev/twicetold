import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd

from between import openai_embed
from settings import settings

MAX_DOC_CHARS = 6000
MIN_DOC_CHARS = 50
EMBED_BATCH = 256
EMBED_THREADS = 2
LABEL_YEAR = 2026
AREA_YEARS = (2023, 2024, 2025, 2026)


def classify(out, data=str(settings.DATA_DIR)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    data, out = Path(data), Path(out)
    papers = pd.read_parquet(data / "papers.parquet")
    sampled = {json.loads(line)["panel_id"] for line in open(data / "panels.jsonl")}
    need = papers[papers.paper_id.isin(sampled) | ((papers.year == LABEL_YEAR) & papers.primary_area.notna())].copy()
    need["doc"] = (need.title.fillna("") + "\n" + need.abstract.fillna("")).str.slice(0, MAX_DOC_CHARS)
    need = need[need.doc.str.len() > MIN_DOC_CHARS].reset_index(drop=True)
    print(len(need), "papers to embed")
    emb = openai_embed(need.doc.tolist(), settings.EMBED_MODEL, EMBED_BATCH, EMBED_THREADS, "embed abstracts")

    labelled = need.primary_area.notna().to_numpy()
    y = need.primary_area[labelled].to_numpy()
    clf = LogisticRegression(max_iter=2000, C=3.0)
    print(f"5-fold accuracy on {LABEL_YEAR} labels:", cross_val_score(clf, emb[labelled], y, cv=5).mean().round(3))
    clf.fit(emb[labelled], y)
    proba = clf.predict_proba(emb)
    pred = clf.classes_[proba.argmax(1)]
    areas = pd.DataFrame(
        dict(
            paper_id=need.paper_id,
            area=np.where(labelled, need.primary_area, pred),
            area_prob=proba.max(1),
            area_pred=pred,
            row=np.arange(len(need)),
        )
    )
    areas.to_parquet(out / "paper_areas.parquet")


def labels(areas, out):
    true_areas = []
    for year in AREA_YEARS:
        d = pd.read_parquet(settings.ICLR_AREAS_DIR / f"mlouis_{year}.parquet", columns=["forum", "primary_area"])
        true_areas.append(d.drop_duplicates("forum")[["forum", "primary_area"]])
    true_areas = pd.concat(true_areas).rename(columns={"forum": "paper_id", "primary_area": "area_true"})
    labelled = pd.read_parquet(areas).merge(true_areas, on="paper_id", how="left")
    labelled["area_final"] = labelled.area_true.fillna(labelled.area)
    labelled["area_source"] = np.where(labelled.area_true.notna(), "openreview", "predicted")
    labelled.to_parquet(out)
    print(labelled.area_source.value_counts().to_dict())


if __name__ == "__main__":
    fire.Fire({"classify": classify, "labels": labels})
