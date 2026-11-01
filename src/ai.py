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
from scipy.sparse import hstack
from scipy.special import logsumexp
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import extract
from outcomes import excluded_review_ids, in_main_sample
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

COCODET_CLASSES = ["human", "mix", "ai"]
COCODET_LOGIT_SCALE = 30.0
COCODET_SORT_CHUNK = 256
COCODET_FLUSH_BATCHES = 250

WORD_HASH = HashingVectorizer(
    ngram_range=(1, 2), n_features=2**18, alternate_sign=False, norm=None, lowercase=True, dtype=np.float32
)
CHAR_HASH = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 4), n_features=2**18, alternate_sign=False, norm=None, lowercase=True, dtype=np.float32
)

SCORE_CHUNK = 5000
DETECTOR_TRAIN_YEARS = [2021, 2022, 2023]
GRADE3_CUTOFF = 0.95
SUMMARY_STRENGTHS_SECTIONS = r"\[(SUMMARY|STRENGTHS)\](.*?)(?=" + TAGS + r"|\Z)"
SUMMARY_STRENGTHS_MIN_CHARS = 200


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
    human_test = pre[~pre.paper_id.isin(train_papers)]
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


def load_cocodet():
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
            self.main_classifier_weight = nn.Parameter(torch.empty(len(COCODET_CLASSES), hidden))
            self.content_classifier = nn.Linear(hidden, 8)
            self.style_classifier = nn.Linear(hidden, 7)
            self.fine_grained_classifier = nn.Linear(hidden, 6)

        def forward(self, input_ids, attention_mask):
            pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            cos = F.linear(F.normalize(pooled, dim=1), F.normalize(self.main_classifier_weight, dim=1))
            return (cos * COCODET_LOGIT_SCALE).softmax(-1)

    model = CoCoDet()
    weights = hf_hub_download(settings.COCODET_REPO, settings.COCODET_WEIGHTS_FILE)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


def cocodet_order(df):
    shuffled = df.assign(text=df.text.str.replace(TAGS + r"\s*", "", regex=True)).sample(frac=1, random_state=0)
    chunk = np.arange(len(shuffled)) // COCODET_SORT_CHUNK
    return shuffled.assign(chunk=chunk, n=shuffled.text.str.len()).sort_values(["chunk", "n"])


def cocodet(input, name=None, max_len=512, batch=8, threads=2, limit=None):
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(threads)
    src = Path(input)
    out = settings.DATA_DIR / f"cocodet_{name or src.stem}.jsonl"
    df = pd.read_parquet(src, columns=["review_id", "text"])
    done = done_ids(out)
    df = df[~df.review_id.isin(done)]
    if limit is not None:
        df = df.head(limit)
    df = cocodet_order(df)
    print(f"{len(done)} done, {len(df)} to score -> {out}")

    tok = AutoTokenizer.from_pretrained(settings.COCODET_BASE_MODEL)
    model = load_cocodet()
    with open(out, "a") as f, torch.inference_mode():
        for batch_no, start in enumerate(tqdm(range(0, len(df), batch), desc="cocodet batches")):
            rows = df.iloc[start : start + batch]
            enc = tok(rows.text.tolist(), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            probs = model(enc["input_ids"], enc["attention_mask"]).tolist()
            for review_id, class_probs in zip(rows.review_id, probs):
                f.write(json.dumps({"review_id": review_id, **dict(zip(COCODET_CLASSES, class_probs))}) + "\n")
            if batch_no % COCODET_FLUSH_BATCHES == 0:
                f.flush()


def style_features(texts):
    cleaned = clean(pd.Series(texts))
    return hstack([WORD_HASH.transform(cleaned), CHAR_HASH.transform(cleaned)]).tocsr()


def p_llm(clf, tfidf, texts, desc):
    chunks = [texts.iloc[start : start + SCORE_CHUNK] for start in range(0, len(texts), SCORE_CHUNK)]
    probs = [clf.predict_proba(tfidf.transform(style_features(chunk)))[:, 1] for chunk in tqdm(chunks, desc=desc)]
    return np.concatenate(probs)


def fit_detector(train):
    features = style_features(train.text)
    tfidf = TfidfTransformer(sublinear_tf=True).fit(features)
    clf = LogisticRegression(C=4.0, max_iter=3000).fit(tfidf.transform(features), train.label)
    return clf, tfidf


def style(out=settings.DATA_DIR / "detector_scores.parquet", by_cell=settings.REPORTS_DIR / "detector_by_cell.csv"):
    intel = pd.read_parquet(settings.DATA_DIR / "intel_reviews.parquet", columns=["label", "llm", "year", "text"])
    train = intel[intel.year.isin(DETECTOR_TRAIN_YEARS)]
    test = intel[intel.year == 2024]
    del intel
    clf, tfidf = fit_detector(train)
    p_test = clf.predict_proba(tfidf.transform(style_features(test.text)))[:, 1]
    print(f"train {len(train)}, test 2024 pairs {len(test)}: AUC {roc_auc_score(test.label, p_test):.4f}")
    for llm_name, g in test.assign(p=p_test).groupby("llm"):
        print(f"  {llm_name:8s} AUC {roc_auc_score(g.label, g.p):.4f}  TPR@p>0.5 {(g[g.label == 1].p > 0.5).mean():.3f}")

    scored = pd.read_parquet(settings.DATA_DIR / "sample_reviews.parquet", columns=["review_id", "conf", "year", "text"])
    scored["p_llm"] = p_llm(clf, tfidf, scored.text, "score reviews")
    pre_period = (scored.conf == "ICLR") & scored.year.between(2017, 2022)
    pre_period_p = scored[pre_period].p_llm
    q90, q99 = np.quantile(pre_period_p, 0.90), np.quantile(pre_period_p, 0.99)
    scored["llm_flag"] = scored.p_llm >= q99
    scored["llm_grade"] = np.select([scored.p_llm < q90, scored.p_llm < q99, scored.p_llm <= GRADE3_CUTOFF], [0, 1, 2], 3)
    print(f"pre-period thresholds: q90={q90:.3f}, q99={q99:.3f}")
    print("share flagged (>= pre-period 99th pct) by venue-year:")
    cell_stats = dict(flag=("llm_flag", "mean"), p_mean=("p_llm", "mean"), g2plus=("llm_grade", lambda g: (g >= 2).mean()))
    in_sample = ~scored.review_id.isin(excluded_review_ids(settings.DATA_DIR))
    by_venue_year = scored[in_sample].groupby(["conf", "year"]).agg(**cell_stats, g3=("llm_grade", lambda g: (g == 3).mean()))
    print(by_venue_year.round(3).to_string())

    has_strengths = scored.text.str.contains("[STRENGTHS]", regex=False)
    has_weaknesses = scored.text.str.contains("[WEAKNESSES]", regex=False)
    combined_strengths_field = has_strengths & ~has_weaknesses
    summary_strengths = (
        scored.text[~combined_strengths_field].str.extractall(SUMMARY_STRENGTHS_SECTIONS)[1].groupby(level=0).apply(" ".join)
    )
    scored["p_llm_ss"] = np.nan
    ss_ids = summary_strengths.index[summary_strengths.str.len() > SUMMARY_STRENGTHS_MIN_CHARS]
    scored.loc[ss_ids, "p_llm_ss"] = p_llm(clf, tfidf, summary_strengths.loc[ss_ids], "score summary+strengths")
    q99_ss = np.quantile(scored[pre_period].p_llm_ss.dropna(), 0.99)
    scored["llm_flag_ss"] = (scored.p_llm_ss >= q99_ss).astype("boolean").where(scored.p_llm_ss.notna())
    print(
        f"summary+strengths exposure available for {len(ss_ids)} reviews, pre-period q99={q99_ss:.3f}; "
        "flagged share by year (ICLR 2024+):"
    )
    iclr_2024_on = in_sample & (scored.conf == "ICLR") & (scored.year >= 2024)
    print(scored[iclr_2024_on].groupby("year").llm_flag_ss.mean().round(3).to_dict())
    scored[["review_id", "p_llm", "llm_flag", "llm_grade", "p_llm_ss", "llm_flag_ss"]].to_parquet(out)

    gptzero = pd.read_parquet(settings.DATA_DIR / "gptzero.parquet", columns=["review_id", "gz_ai", "gz_isAI"])
    with_gptzero = scored.merge(gptzero, on="review_id")
    for year in (2023, 2024):
        g = with_gptzero[with_gptzero.year == year]
        print(
            f"ICLR {year}: GPTZero isAI {g.gz_isAI.mean():.3f}, ours flagged {g.llm_flag.mean():.3f}; "
            f"AUC of p_llm for GPTZero isAI {roc_auc_score(g.gz_isAI, g.p_llm):.3f}; "
            f"corr(p_llm, gz_ai) {np.corrcoef(g.p_llm, g.gz_ai)[0, 1]:.3f}; "
            f"P(flag|isAI) {g[g.gz_isAI].llm_flag.mean():.3f}, P(isAI|flag) {g[g.llm_flag].gz_isAI.mean():.3f}"
        )
    scored[in_sample].groupby(["conf", "year"]).agg(**cell_stats).round(4).to_csv(by_cell)


if __name__ == "__main__":
    fire.Fire(
        {"reference": reference, "wordshare": wordshare, "wordshare_fields": wordshare_fields, "cocodet": cocodet, "style": style}
    )
