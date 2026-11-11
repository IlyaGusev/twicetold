import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from openai import APIError, BadRequestError, OpenAI, RateLimitError
from tqdm import tqdm

import extract
import hub
from outcomes import main_sample_review_ids
from settings import settings

AREA_PAIRS = 300
CELL_PAIRS = 3000
EMBED_ATTEMPTS = 40
EMBED_MAX_BACKOFF_S = 30
REVIEW_EMBED_CHARS = 20000
REVIEW_EMBED_SLICE = 20000


def openai_embed(texts, model, batch, threads, desc, client=None, **options):
    client = client or OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2)

    def embed_chunk(chunk):
        for attempt in range(EMBED_ATTEMPTS):
            try:
                response = client.embeddings.create(model=model, input=chunk, **options)
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


def claims_table(model=settings.DEFAULT_MODEL, data=settings.WORK_DIR):
    data = Path(data)
    reviews = hub.corpus_reviews(["review_id", "paper_id", "conf", "year"]).set_index("review_id")
    rows = []
    for review_id, claims in tqdm(hub.claims(model).items(), desc="claims table"):
        if review_id not in reviews.index:
            continue
        review = reviews.loc[review_id]
        for k, claim in enumerate(claims):
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


def embed_reviews(data, threads, reembed):
    path = data / "review_emb.npy"
    R = hub.sample("main", ["review_id", "text"])
    done = None if reembed or not path.exists() else np.load(path)
    client = extract.openrouter_client()
    while done is None or len(done) < len(R):
        start = 0 if done is None else len(done)
        texts = [t[:REVIEW_EMBED_CHARS] for t in R.text.iloc[start : start + REVIEW_EMBED_SLICE].fillna("")]
        E = openai_embed(
            texts,
            settings.REVIEW_EMBED_MODEL,
            batch=64,
            threads=threads,
            desc=f"embed reviews from {start}",
            client=client,
            dimensions=settings.REVIEW_EMBED_DIMENSIONS,
        )
        done = E if done is None else np.concatenate([done, E])
        np.save(path, done)


def load_reviews(data):
    R = hub.sample("main", ["conf", "year", "paper_id", "review_id"])
    R["row"] = np.arange(len(R))
    R = R[R.review_id.isin(main_sample_review_ids())].merge(hub.paper_areas(), on="paper_id", how="left")
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


def embed(
    model=settings.DEFAULT_MODEL, reembed=False, threads=4, data=settings.ADDITIONAL_DIR, reports=settings.ANALYSIS_REPORTS_DIR
):
    data, reports = Path(data), Path(reports)
    embed_reviews(data, threads, reembed)
    R, E = load_reviews(data)
    intra = intra_paper(R, E)
    cells = cell_similarity(R, E, intra, np.random.default_rng(0))
    cells.to_csv(reports / f"embed_{model}.csv", index=False)
    print(cells.round(4).to_string(index=False))


if __name__ == "__main__":
    fire.Fire({"embed": embed, "claims_table": claims_table})
