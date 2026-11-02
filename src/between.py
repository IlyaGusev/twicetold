import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from openai import APIError, BadRequestError, OpenAI, RateLimitError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcomes import main_sample_review_ids
from settings import settings

AREA_PAIRS = 300
CELL_PAIRS = 3000
EMBED_ATTEMPTS = 40
EMBED_MAX_BACKOFF_S = 30
REVIEW_EMBED_CHARS = 12000


def openai_embed(texts, model, batch, threads, desc):
    client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2)

    def embed_chunk(chunk):
        for attempt in range(EMBED_ATTEMPTS):
            try:
                response = client.embeddings.create(model=model, input=chunk)
                return np.array([item.embedding for item in response.data], dtype=np.float32)
            except BadRequestError:
                chunk = [t[: len(t) // 2] for t in chunk]
            except RateLimitError, APIError:
                time.sleep(min(EMBED_MAX_BACKOFF_S, 2**attempt))
        raise RuntimeError("embedding batch failed after retries")

    chunks = [texts[i : i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(threads) as pool:
        embedded = tqdm(pool.map(embed_chunk, chunks), total=len(chunks), desc=desc)
        return np.concatenate(list(embedded))


def claims_table(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    data = Path(data)
    reviews = pd.read_parquet(data / "reviews.parquet", columns=["review_id", "paper_id", "conf", "year"]).set_index("review_id")
    rows = []
    with open(data / f"claims_{model}.jsonl") as f:
        for line in tqdm(f, desc="claims table"):
            record = json.loads(line)
            if "claims" not in record or record["review_id"] not in reviews.index:
                continue
            review_id = record["review_id"]
            review = reviews.loc[review_id]
            for k, claim in enumerate(record["claims"]):
                rows.append(
                    {
                        "review_id": review_id,
                        "k": k,
                        "paper_id": review.paper_id,
                        "conf": review.conf,
                        "year": int(review.year),
                        "claim": claim["claim"],
                        "category": claim["category"],
                        "target": claim["target"],
                        "specific": claim["specific"],
                    }
                )
    pd.DataFrame(rows).to_parquet(data / f"claims_table_{model}.parquet", index=False)


def normalize_rows(X):
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X


def embed_reviews(data, threads):
    R = pd.read_parquet(data / "sample_reviews.parquet", columns=["review_id", "text"])
    texts = [t[:REVIEW_EMBED_CHARS] for t in R.text.fillna("")]
    E = openai_embed(texts, settings.EMBED_MODEL, batch=64, threads=threads, desc="embed reviews")
    np.save(data / "review_emb.npy", E)
    print("embedded", len(texts))


def load_reviews(data):
    R = pd.read_parquet(data / "sample_reviews.parquet", columns=["conf", "year", "paper_id", "review_id"])
    R["row"] = np.arange(len(R))
    areas = pd.read_parquet(data / "paper_areas.parquet", columns=["paper_id", "area_pred"])
    R = R[R.review_id.isin(main_sample_review_ids(data))].merge(areas, on="paper_id", how="left")
    E = normalize_rows(np.load(data / "review_emb.npy"))
    return R, E


def intra_paper(R, E):
    rows = []
    for paper_id, paper in tqdm(R.groupby("paper_id"), desc="intra-paper"):
        if len(paper) < 2:
            continue
        emb_rows = paper.row.to_numpy()
        sims = E[emb_rows] @ E[emb_rows].T
        pair_sims = sims[np.triu_indices(len(paper), 1)]
        rows.append(
            {"conf": paper.conf.iloc[0], "year": paper.year.iloc[0], "paper_id": paper_id, "intra": float(pair_sims.mean())}
        )
    return pd.DataFrame(rows)


def cross_paper_sims(reviews, E, rng, n_pairs):
    emb_rows = reviews.row.to_numpy()
    papers = reviews.paper_id.to_numpy()
    sims = []
    for _ in range(n_pairs):
        a, b = rng.choice(len(emb_rows), 2, replace=False)
        if papers[a] != papers[b]:
            sims.append(float(E[emb_rows[a]] @ E[emb_rows[b]]))
    return sims


def cell_similarity(R, E, intra, rng):
    rows = []
    for (conf, year), cell in tqdm(R.groupby(["conf", "year"]), desc="inter-paper"):
        area_sims = []
        for _, area in cell.groupby(cell.area_pred.fillna("none")):
            if area.paper_id.nunique() >= 2:
                area_sims += cross_paper_sims(area, E, rng, AREA_PAIRS)
        any_sims = cross_paper_sims(cell, E, rng, CELL_PAIRS)
        cell_intra = intra[(intra.conf == conf) & (intra.year == year)].intra
        rows.append(
            {
                "conf": conf,
                "year": year,
                "intra": cell_intra.mean(),
                "intra_se": cell_intra.std(ddof=1) / np.sqrt(len(cell_intra)),
                "inter_area": np.mean(area_sims) if area_sims else np.nan,
                "inter_any": np.mean(any_sims),
                "papers": len(cell_intra),
            }
        )
    cells = pd.DataFrame(rows)
    cells["excess"] = cells.intra - cells.inter_area
    return cells


def embed(model=settings.DEFAULT_MODEL, reembed=False, threads=4, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    data, reports = Path(data), Path(reports)
    if reembed:
        embed_reviews(data, threads)
    R, E = load_reviews(data)
    intra = intra_paper(R, E)
    cells = cell_similarity(R, E, intra, np.random.default_rng(0))
    cells.to_csv(reports / f"embed_{model}.csv", index=False)
    print(cells.round(4).to_string(index=False))


if __name__ == "__main__":
    fire.Fire({"embed": embed, "claims_table": claims_table})
