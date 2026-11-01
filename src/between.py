import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import settings

K_TYPES = 300
REF_PER_CELL = 10000
SIM_BLOCK_ROWS = 4096
KMEANS_FIT_SAMPLE = 300000
PREDICT_CHUNK_ROWS = 200000
HILL_SAMPLE = 20000
TEMPLATE_QUANTILE = 0.75
SHIFT_ROWS_SHOWN = 12
AREA_PAIRS = 300
CELL_PAIRS = 3000
EMBED_ATTEMPTS = 40
EMBED_MAX_BACKOFF_S = 30
CLAIM_EMBED_CHARS = 400
CLAIM_EMBED_DIMS = 512
REVIEW_EMBED_CHARS = 12000


def openai_embed(texts, model, batch, threads=4, dtype=np.float32, desc=None, **kw):
    from openai import APIError, BadRequestError, OpenAI, RateLimitError

    client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2)

    def embed_chunk(chunk):
        for attempt in range(EMBED_ATTEMPTS):
            try:
                response = client.embeddings.create(model=model, input=chunk, **kw)
                return np.array([item.embedding for item in response.data], dtype=dtype)
            except BadRequestError:
                chunk = [t[: len(t) // 2] for t in chunk]
            except RateLimitError, APIError:
                time.sleep(min(EMBED_MAX_BACKOFF_S, 2**attempt))
        raise RuntimeError("embedding batch failed after retries")

    chunks = [texts[i : i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(threads) as pool:
        embedded = tqdm(pool.map(embed_chunk, chunks), total=len(chunks), desc=desc, disable=desc is None)
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
    return pd.DataFrame(rows)


def write_claims_table(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    claims_table(model, data).to_parquet(Path(data) / f"claims_table_{model}.parquet", index=False)


def embed_claims(model, data):
    T = claims_table(model, data)
    T["emb_row"] = np.arange(len(T))
    T.to_parquet(data / f"claims_table_{model}.parquet")
    E = openai_embed(
        T.claim.str.slice(0, CLAIM_EMBED_CHARS).tolist(),
        settings.CLAIM_EMBED_MODEL,
        batch=512,
        dtype=np.float16,
        desc="embed claims",
        dimensions=CLAIM_EMBED_DIMS,
    )
    np.save(data / f"claim_emb_{model}.npy", E)


def normalize_rows(X):
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X


def load_claims(model, data):
    T = pd.read_parquet(data / f"claims_table_{model}.parquet")
    review_ids = set(pd.read_parquet(data / "reviews.parquet", columns=["review_id"]).review_id)
    T = T[T.review_id.isin(review_ids)].reset_index(drop=True)
    E = np.load(data / f"claim_emb_{model}.npy", mmap_mode="r")[T.emb_row.to_numpy()].astype(np.float32)
    return T, normalize_rows(E)


def template_similarity(T, E, rng):
    paper_ids = T.paper_id.to_numpy()
    best_sim = np.full(len(T), np.nan)
    for cell_rows in tqdm(T.groupby(["conf", "year"]).indices.values(), desc="template sim"):
        ref_rows = rng.choice(cell_rows, min(REF_PER_CELL, len(cell_rows)), replace=False)
        ref_emb = E[ref_rows].T
        for start in range(0, len(cell_rows), SIM_BLOCK_ROWS):
            block = cell_rows[start : start + SIM_BLOCK_ROWS]
            sim = E[block] @ ref_emb
            same_paper = paper_ids[block][:, None] == paper_ids[ref_rows][None, :]
            sim[same_paper] = -1
            best_sim[block] = sim.max(1)
    return best_sim


def fit_types(E, pre_rows, rng):
    from sklearn.cluster import MiniBatchKMeans

    fit_rows = rng.choice(pre_rows, min(KMEANS_FIT_SAMPLE, len(pre_rows)), replace=False)
    km = MiniBatchKMeans(K_TYPES, random_state=0, batch_size=4096, n_init=3).fit(E[fit_rows])
    chunk_starts = range(0, len(E), PREDICT_CHUNK_ROWS)
    ctype = np.concatenate([km.predict(E[s : s + PREDICT_CHUNK_ROWS]) for s in chunk_starts])
    return km, ctype


def type_labels(km, pre, E):
    unit_centers = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    labels = {}
    for c, members in pre.groupby("ctype").groups.items():
        closest = members[np.argmax(E[members] @ unit_centers[c])]
        labels[c] = pre.claim[closest]
    return pd.Series(labels, name="label")


def cell_summary(T, rng):
    by_cell = T.groupby(["conf", "year"])
    summary = by_cell.agg(
        claims=("claim", "size"),
        template_sim=("template_sim", "mean"),
        template_share=("template", "mean"),
        tail_share=("tail_type", "mean"),
        specific=("specific", "mean"),
    )
    hill = {}
    for key, cell_rows in by_cell.indices.items():
        sample = rng.choice(cell_rows, min(HILL_SAMPLE, len(cell_rows)), replace=False)
        type_share = T.ctype.iloc[sample].value_counts(normalize=True)
        hill[key] = float(np.exp(-(type_share * np.log(type_share)).sum()))
    summary["hill_types_20k"] = pd.Series(hill)
    return summary


def type_shift(T, labels):
    iclr = T[T.conf == "ICLR"]
    freq = pd.crosstab(iclr.ctype, iclr.year, normalize="columns")
    delta = (freq[2026] - freq[2023]).sort_values()
    shift = pd.DataFrame({"label": labels, "f2023": freq[2023], "f2026": freq[2026], "delta": delta})
    return shift.reindex(delta.index)


def templates(model=settings.DEFAULT_MODEL, reembed=False, data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    data, reports = Path(data), Path(reports)
    if reembed:
        embed_claims(model, data)
    T, E = load_claims(model, data)
    rng = np.random.default_rng(0)

    T["template_sim"] = template_similarity(T, E, rng)
    is_pre = (T.conf == "ICLR") & T.year.between(2017, 2022)
    tau = np.quantile(T.template_sim[is_pre], TEMPLATE_QUANTILE)
    T["template"] = T.template_sim >= tau
    print(f"template threshold tau={tau:.3f} (75th percentile of ICLR 2017-2022)")

    km, ctype = fit_types(E, np.flatnonzero(is_pre), rng)
    T["ctype"] = ctype
    pre = T[is_pre]
    freq_pre = pre.ctype.value_counts(normalize=True)
    rank = freq_pre.rank(ascending=False)
    tail_types = set(rank[rank > K_TYPES / 2].index)
    T["tail_type"] = T.ctype.isin(tail_types)
    labels = type_labels(km, pre, E)
    critique_types = labels.to_frame()
    critique_types["freq_pre"] = freq_pre
    critique_types["tail"] = critique_types.index.isin(tail_types)
    critique_types.to_csv(reports / f"critique_types_{model}.csv")
    T.to_parquet(data / f"claims_table_{model}.parquet")

    cells = cell_summary(T, rng)
    cells.round(4).to_csv(reports / f"template_by_cell_{model}.csv")
    print(cells.round(3).to_string())

    shift = type_shift(T, labels)
    shift.to_csv(reports / f"type_shift_{model}.csv")
    print("\nBiggest falls 2023->2026:\n", shift.head(SHIFT_ROWS_SHOWN).round(4).to_string())
    print("\nBiggest rises 2023->2026:\n", shift.tail(SHIFT_ROWS_SHOWN).round(4).to_string())


def embed_reviews(data, threads):
    R = pd.read_parquet(data / "sample_reviews.parquet", columns=["review_id", "text"])
    texts = [t[:REVIEW_EMBED_CHARS] for t in R.text.fillna("")]
    E = openai_embed(texts, settings.EMBED_MODEL, batch=64, threads=threads, desc="embed reviews")
    np.save(data / "review_emb.npy", E)
    print("embedded", len(texts))


def load_reviews(data):
    R = pd.read_parquet(data / "sample_reviews.parquet", columns=["conf", "year", "paper_id", "review_id"])
    R["row"] = np.arange(len(R))
    areas = pd.read_parquet(data / "reviews.parquet", columns=["review_id", "primary_area"]).drop_duplicates("review_id")
    R = R.merge(areas, on="review_id")
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
        for _, area in cell.groupby(cell.primary_area.fillna("none")):
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


def year_effects(intra):
    import statsmodels.formula.api as smf

    fit = smf.ols("intra ~ C(year, Treatment(reference=2023))", data=intra[intra.conf == "ICLR"]).fit(cov_type="HC1")
    print("\nICLR IntraSim year effects vs 2023:")
    for term, coef in fit.params.items():
        if "year" in term:
            year = term.split("T.")[1].rstrip("]")
            print(f"  {year:6s} {coef:+.4f} (SE {fit.bse[term]:.4f})")


def claim_vs_embedding(intra, model, data):
    outcomes = ["S3", "share_shared_claims", "D3"]
    panels = pd.read_parquet(data / f"panel_outcomes_{model}.parquet", columns=["paper_id", *outcomes])
    merged = intra.merge(panels, on="paper_id")
    rho = {outcome: merged[["intra", outcome]].corr("spearman").iloc[0, 1] for outcome in outcomes}
    print(
        f"\npaper-level Spearman: intra vs S3 {rho['S3']:.3f}, vs share shared {rho['share_shared_claims']:.3f}, "
        f"vs D3 {rho['D3']:.3f} (n={len(merged)})"
    )


def embed(model=settings.DEFAULT_MODEL, reembed=False, threads=4, data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    data, reports = Path(data), Path(reports)
    if reembed:
        embed_reviews(data, threads)
    R, E = load_reviews(data)
    intra = intra_paper(R, E)
    cells = cell_similarity(R, E, intra, np.random.default_rng(0))
    cells.to_csv(reports / f"embed_{model}.csv", index=False)
    print(cells.round(4).to_string(index=False))
    year_effects(intra)
    claim_vs_embedding(intra, model, data)


if __name__ == "__main__":
    fire.Fire({"templates": templates, "embed": embed, "claims_table": write_claims_table})
