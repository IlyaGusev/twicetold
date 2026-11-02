import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

import extract
from outcomes import in_main_sample
from prompts import render
from settings import settings

SECTION_TAGS = [
    "SUMMARY",
    "MAIN",
    "STRENGTHS",
    "WEAKNESSES",
    "QUESTIONS",
    "LIMITATIONS",
    "OTHER",
    "STRENGTHS AND WEAKNESSES",
    "CLAIMS AND EVIDENCE",
    "METHODS AND EVALUATION",
    "THEORY",
    "EXPERIMENTS",
    "SUPPLEMENT",
    "RELATED WORK",
    "MISSING REFERENCES",
    "REPORT",
    "REQUESTED CHANGES",
]
TAGS = r"\[(?:" + "|".join(SECTION_TAGS) + r")\]"

OUTLINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["points"],
    "properties": {"points": {"type": "array", "items": {"type": "string"}}},
}
REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["review"],
    "properties": {"review": {"type": "string"}},
}
VENUE = {
    "ICLR": "a machine-learning conference (ICLR)",
    "NeurIPS": "a machine-learning conference (NeurIPS)",
    "SciPost": "a physics journal (SciPost)",
    "F1000": "an open-research biomedical journal (F1000Research)",
    "BMC-Med": "a medical journal (BMC Medicine)",
    "PLOS": "a multidisciplinary journal (PLOS ONE)",
    "Copernicus": "a geoscience journal (Copernicus/EGU)",
}

WORDSHARE_HUMAN_YEARS = (2018, 2022)


def clean(s):
    return s.str.replace(TAGS, " ", regex=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def done_ids(path):
    return {r["review_id"] for r in read_jsonl(path)} if path.exists() else set()


def load_ml_reviews(columns):
    reviews = pd.read_parquet(
        settings.DATA_DIR / "reviews.parquet", columns=[*columns, "withdrawn", "decision"], filters=[("withdrawn", "==", False)]
    )
    return reviews[in_main_sample(reviews)][columns]


def reference_sample(per_source):
    cols = ["conf", "year", "review_id", "text"]
    ml = pd.read_parquet(
        settings.DATA_DIR / "reviews.parquet", columns=cols, filters=[("year", ">=", 2018), ("year", "<=", 2021)]
    )
    other = pd.read_parquet(settings.DATA_DIR / "other_reviews.parquet", columns=cols, filters=[("year", "<=", 2021)])
    return pd.concat([ml, other]).sample(frac=1, random_state=0).groupby("conf").head(per_source)


def rewrite(row, model, effort):
    outline_prompt = render("ai_outline")
    outline_reply = extract.call(
        outline_prompt, row.text[: settings.REVIEW_TEXT_MAX_CHARS], OUTLINE_SCHEMA, "outline", model=model, effort=effort
    )
    if "error" in outline_reply:
        return None
    outline = "\n".join(f"- {point}" for point in outline_reply["points"])
    words = round(len(row.text.split()), -1)
    write_prompt = render("ai_write", venue=VENUE[row.conf], words=words)
    review_reply = extract.call(write_prompt, outline, REVIEW_SCHEMA, "review", model=model, effort=effort)
    if "error" in review_reply:
        return None
    return {
        "review_id": row.review_id,
        "conf": row.conf,
        "year": int(row.year),
        "outline": outline,
        "text": review_reply["review"],
    }


def reference(model=settings.AI_REFERENCE_MODEL, threads=32, per_source=3000, out=None):
    effort = extract.reasoning_efforts(model).pass1
    out = Path(out) if out else settings.DATA_DIR / f"ai_reference_{model}.jsonl"
    done = done_ids(out)
    todo = [r for r in reference_sample(per_source).itertuples() if r.review_id not in done]
    print(f"{len(done)} done, {len(todo)} to write -> {out}")
    with open(out, "a") as f, ThreadPoolExecutor(threads) as pool:
        bar = tqdm(pool.map(partial(rewrite, model=model, effort=effort), todo), total=len(todo), desc="reference")
        for res in bar:
            if res:
                f.write(json.dumps(res) + "\n")
            bar.set_postfix(cost=f"${extract.cost():.2f}")
    print(f"done, ${extract.cost():.2f}")


def half_of_papers(paper_ids, rng):
    papers = paper_ids.dropna().unique()
    return set(rng.choice(papers, len(papers) // 2, replace=False))


def presence_rate(vec, texts):
    X = vec.transform(clean(texts))
    return (np.asarray(X.sum(0)).ravel() + 1) / (X.shape[0] + 2)


def fit_rates(human, ai, n_fit):
    vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b[a-z][a-z]+\b", min_df=0.01, max_df=0.5)
    fit_texts = pd.concat([clean(s.sample(min(n_fit, len(s)), random_state=0)) for s in (human, ai)])
    vec.fit(fit_texts)
    return vec, presence_rate(vec, human), presence_rate(vec, ai)


def doc_loglik(X, rate):
    log_present = np.log(rate)
    log_absent = np.log1p(-rate)
    return X @ (log_present - log_absent) + log_absent.sum()


def loglik(rates, texts):
    vec, human_rate, ai_rate = rates
    X = vec.transform(clean(texts))
    return doc_loglik(X, human_rate), doc_loglik(X, ai_rate)


def mle_alpha(lp, lq):
    def nll(a):
        return -logsumexp(np.stack([lp + np.log1p(-a), lq + np.log(a)]), axis=0).sum()

    return minimize_scalar(nll, bounds=(1e-6, 1 - 1e-6), method="bounded").x


def estimate(rates, texts, rng, n_boot=200):
    lp, lq = loglik(rates, texts)
    boot_samples = rng.integers(0, len(lp), (n_boot, len(lp)))
    boot_alphas = [mle_alpha(lp[sample], lq[sample]) for sample in boot_samples]
    lo, hi = np.percentile(boot_alphas, [2.5, 97.5])
    return {"alpha": mle_alpha(lp, lq), "lo": lo, "hi": hi}


def recovered(human, ai, alpha, n, rng):
    human_lp, human_lq = human
    ai_lp, ai_lq = ai
    human_idx = rng.choice(len(human_lp), int(n * (1 - alpha)), replace=False)
    ai_idx = rng.choice(len(ai_lp), int(n * alpha), replace=False)
    return mle_alpha(np.r_[human_lp[human_idx], ai_lp[ai_idx]], np.r_[human_lq[human_idx], ai_lq[ai_idx]])


def wordshare(out=settings.REPORTS_DIR / "llm_share.csv"):
    rng = np.random.default_rng(0)
    cols = ["conf", "year", "paper_id", "review_id", "text"]
    reviews = pd.concat(
        [load_ml_reviews(cols), pd.read_parquet(settings.DATA_DIR / "icml_sample.parquet", columns=cols)], ignore_index=True
    )
    llm = pd.read_parquet(settings.DATA_DIR / "intel_reviews.parquet", columns=["label", "text", "paper_id"]).query("label == 1")

    first_year, last_year = WORDSHARE_HUMAN_YEARS
    pre = reviews[reviews.year.between(first_year, last_year)]
    train_papers = half_of_papers(pre.paper_id, rng)
    held_out_papers = set(pre.paper_id) - train_papers
    human_train = pre[pre.paper_id.isin(train_papers)]
    human_test = pre[pre.paper_id.isin(held_out_papers)]
    llm_train = llm[llm.paper_id.isin(train_papers)]
    llm_test = llm[llm.paper_id.isin(held_out_papers)]
    rates = fit_rates(human_train.text, llm_train.text, 20000)
    vec, _, _ = rates
    print(f"vocabulary {len(vec.vocabulary_)} words; human train {len(human_train)}, LLM train {len(llm_train)}")

    human_test_ll = loglik(rates, human_test.text)
    llm_test_ll = loglik(rates, llm_test.text)
    print("Recovery on held-out mixtures (true -> estimated):")
    for a in [0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        print(f"  {a:.2f} -> {recovered(human_test_ll, llm_test_ll, a, 4000, rng):.3f}")

    rows = []
    for (conf, year), venue_year in tqdm(reviews.groupby(["conf", "year"]), desc="venue-years"):
        held_out = venue_year[~venue_year.paper_id.isin(train_papers)]
        share = estimate(rates, held_out.text, rng)
        rows.append(dict(conf=conf, year=year, n=len(held_out), **share, in_reference=first_year <= year <= last_year))
    out_df = pd.DataFrame(rows)
    print(out_df.round(3).to_string(index=False))
    out_df.to_csv(out, index=False)


def wordshare_fields(
    reference=settings.DATA_DIR / f"ai_reference_{settings.AI_REFERENCE_MODEL}.jsonl",
    out=settings.REPORTS_DIR / "llm_share_all.csv",
):
    rng = np.random.default_rng(1)
    boot_rng = np.random.default_rng(0)
    cols = ["conf", "year", "paper_id", "review_id", "text"]
    reviews = pd.concat(
        [load_ml_reviews(cols), pd.read_parquet(settings.DATA_DIR / "other_reviews.parquet", columns=cols)], ignore_index=True
    )
    ai_reviews = pd.DataFrame(read_jsonl(reference)).merge(reviews[["review_id", "paper_id"]], on="review_id")

    rows = []
    for conf, source_reviews in tqdm(reviews.groupby("conf"), desc="sources"):
        source_ai = ai_reviews[ai_reviews.conf == conf]
        pre = source_reviews[source_reviews.year <= 2021]
        train_papers = half_of_papers(pre.paper_id, rng)
        human_train = pre[pre.paper_id.isin(train_papers)]
        human_test = pre[~pre.paper_id.isin(train_papers)]
        ai_train = source_ai[source_ai.paper_id.isin(train_papers)]
        ai_test = source_ai[~source_ai.paper_id.isin(train_papers)]
        rates = fit_rates(human_train.text, ai_train.text, 5000)
        vec, _, _ = rates
        human_test_ll = loglik(rates, human_test.text)
        ai_test_ll = loglik(rates, ai_test.text)
        recovery = []
        for a in [0.1, 0.3]:
            n = min(2000, len(human_test), int(len(ai_test) / a))
            recovery.append(f"{a:.1f}->{recovered(human_test_ll, ai_test_ll, a, n, rng):.3f}")
        tqdm.write(
            f"{conf}: vocab {len(vec.vocabulary_)}, human train {len(human_train)}, AI train {len(ai_train)}, "
            f"recovery {' '.join(recovery)}"
        )
        for year, venue_year in source_reviews.groupby("year"):
            held_out = venue_year[~venue_year.paper_id.isin(train_papers)]
            share = estimate(rates, held_out.text, boot_rng)
            rows.append(dict(conf=conf, year=year, n=len(held_out), **share))
    out_df = pd.DataFrame(rows)
    print(out_df.round(3).to_string(index=False))
    out_df.to_csv(out, index=False)


if __name__ == "__main__":
    fire.Fire({"reference": reference, "wordshare": wordshare, "wordshare_fields": wordshare_fields})
