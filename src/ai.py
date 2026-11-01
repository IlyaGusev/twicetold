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
from prompts import render
from settings import settings

TAGS = r"\[(SUMMARY|MAIN|STRENGTHS|WEAKNESSES|QUESTIONS|LIMITATIONS|OTHER)\]"

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

COCODET_BASE = "answerdotai/ModernBERT-base"
COCODET_CLASSES = ["human", "mix", "ai"]

WORD_HASH = HashingVectorizer(
    ngram_range=(1, 2), n_features=2**18, alternate_sign=False, norm=None, lowercase=True, dtype=np.float32
)
CHAR_HASH = HashingVectorizer(
    analyzer="char_wb", ngram_range=(3, 4), n_features=2**18, alternate_sign=False, norm=None, lowercase=True, dtype=np.float32
)


def clean(s):
    return s.str.replace(TAGS, " ", regex=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def done_ids(path):
    return {r["review_id"] for r in read_jsonl(path)} if path.exists() else set()


def load_ml(columns):
    return pd.read_parquet(settings.DATA_DIR / "reviews.parquet", columns=columns, filters=[("withdrawn", "==", False)])


def reference_sample(per_source):
    cols = ["conf", "year", "review_id", "text"]
    ml = pd.read_parquet(
        settings.DATA_DIR / "reviews.parquet", columns=cols, filters=[("year", ">=", 2018), ("year", "<=", 2021)]
    )
    other = pd.read_parquet(settings.DATA_DIR / "other_reviews.parquet", columns=cols, filters=[("year", "<=", 2021)])
    return pd.concat([ml, other]).sample(frac=1, random_state=0).groupby("conf").head(per_source)


def rewrite(row, model):
    o = extract.call(render("ai_outline"), row.text[: settings.REVIEW_TEXT_MAX_CHARS], OUTLINE_SCHEMA, "outline", model=model)
    if "error" in o:
        return None
    outline = "\n".join(f"- {p}" for p in o["points"])
    words = round(len(row.text.split()), -1)
    w = extract.call(render("ai_write", venue=VENUE[row.conf], words=words), outline, REVIEW_SCHEMA, "review", model=model)
    if "error" in w:
        return None
    return {"review_id": row.review_id, "conf": row.conf, "year": int(row.year), "outline": outline, "text": w["review"]}


def reference(model=settings.DEFAULT_MODEL, threads=32, per_source=3000, out=settings.DATA_DIR / "ai_reference.jsonl"):
    out = Path(out)
    done = done_ids(out)
    todo = [r for r in reference_sample(per_source).itertuples() if r.review_id not in done]
    print(f"{len(done)} done, {len(todo)} to write -> {out}")
    with open(out, "a") as f, ThreadPoolExecutor(threads) as pool:
        bar = tqdm(pool.map(partial(rewrite, model=model), todo), total=len(todo), desc="reference")
        for res in bar:
            if res:
                f.write(json.dumps(res) + "\n")
            bar.set_postfix(cost=f"${extract.cost():.2f}")
    print(f"done, ${extract.cost():.2f}")


def doc_loglik(X, rate):
    lp, lq = np.log(rate), np.log1p(-rate)
    return X @ (lp - lq) + lq.sum()


def mle_alpha(lp, lq):
    def nll(a):
        return -logsumexp(np.stack([lp + np.log1p(-a), lq + np.log(a)]), axis=0).sum()

    return minimize_scalar(nll, bounds=(1e-6, 1 - 1e-6), method="bounded").x


def estimate(rates, texts, rng, n_boot=200):
    lp, lq = loglik(rates, texts)
    idx = rng.integers(0, len(lp), (n_boot, len(lp)))
    lo, hi = np.percentile([mle_alpha(lp[i], lq[i]) for i in idx], [2.5, 97.5])
    return {"alpha": mle_alpha(lp, lq), "lo": lo, "hi": hi}


def half(paper_ids, rng):
    papers = paper_ids.unique()
    return set(rng.choice(papers, len(papers) // 2, replace=False))


def fit_rates(human, ai, n_fit):
    def rates(texts):
        X = vec.transform(clean(texts))
        return (np.asarray(X.sum(0)).ravel() + 1) / (X.shape[0] + 2)

    vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b[a-z][a-z]+\b", min_df=0.01, max_df=0.5)
    vec.fit(pd.concat([clean(s.sample(min(n_fit, len(s)), random_state=0)) for s in (human, ai)]))
    return vec, rates(human), rates(ai)


def loglik(rates, texts):
    vec, p, q = rates
    X = vec.transform(clean(texts))
    return doc_loglik(X, p), doc_loglik(X, q)


def recovered(human, ai, alpha, n, rng):
    (lh, lqh), (ll, lql) = human, ai
    ih = rng.choice(len(lh), int(n * (1 - alpha)), replace=False)
    il = rng.choice(len(ll), int(n * alpha), replace=False)
    return mle_alpha(np.r_[lh[ih], ll[il]], np.r_[lqh[ih], lql[il]])


def wordshare(out=settings.REPORTS_DIR / "llm_share.csv"):
    rng = np.random.default_rng(0)
    cols = ["conf", "year", "paper_id", "review_id", "text"]
    reviews = pd.concat(
        [load_ml(cols), pd.read_parquet(settings.DATA_DIR / "icml_sample.parquet", columns=cols)], ignore_index=True
    )
    llm = pd.read_parquet(settings.DATA_DIR / "intel_reviews.parquet", columns=["label", "text", "paper_id"]).query("label == 1")

    pre = reviews[reviews.year.between(2018, 2022)]
    train = half(pre.paper_id, rng)
    human_train = pre[pre.paper_id.isin(train)]
    llm_train = half(llm.paper_id, rng)
    llm_train, llm_test = llm[llm.paper_id.isin(llm_train)], llm[~llm.paper_id.isin(llm_train)]
    rates = fit_rates(human_train.text, llm_train.text, 20000)
    print(f"vocabulary {len(rates[1])} words; human train {len(human_train)}, LLM train {len(llm_train)}")

    held, llm_held = loglik(rates, pre[~pre.paper_id.isin(train)].text), loglik(rates, llm_test.text)
    print("Recovery on held-out mixtures (true -> estimated):")
    for a in [0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        print(f"  {a:.2f} -> {recovered(held, llm_held, a, 4000, rng):.3f}")

    rows = []
    for (conf, year), g in tqdm(reviews.groupby(["conf", "year"]), desc="venue-years"):
        g = g[~g.paper_id.isin(train)]
        rows.append(dict(conf=conf, year=year, n=len(g), **estimate(rates, g.text, rng), in_reference=2018 <= year <= 2022))
    out_df = pd.DataFrame(rows)
    print(out_df.round(3).to_string(index=False))
    out_df.to_csv(out, index=False)


def wordshare_fields(reference=settings.DATA_DIR / "ai_reference.jsonl", out=settings.REPORTS_DIR / "llm_share_all.csv"):
    rng, boot_rng = np.random.default_rng(1), np.random.default_rng(0)
    cols = ["conf", "year", "paper_id", "review_id", "text"]
    reviews = pd.concat(
        [load_ml(cols), pd.read_parquet(settings.DATA_DIR / "other_reviews.parquet", columns=cols)], ignore_index=True
    )
    ai = pd.DataFrame(read_jsonl(reference)).merge(reviews[["review_id", "paper_id"]], on="review_id")

    rows = []
    for conf, R in tqdm(reviews.groupby("conf"), desc="sources"):
        Q = ai[ai.conf == conf]
        pre = R[R.year <= 2021]
        train = half(pre.paper_id, rng)
        h_train, q_train, q_test = pre[pre.paper_id.isin(train)], Q[Q.paper_id.isin(train)], Q[~Q.paper_id.isin(train)]
        rates = fit_rates(h_train.text, q_train.text, 5000)
        held, ai_held = loglik(rates, pre[~pre.paper_id.isin(train)].text), loglik(rates, q_test.text)
        rec = []
        for a in [0.1, 0.3]:
            n = min(2000, len(held[0]), int(len(q_test) / a))
            rec.append(f"{a:.1f}->{recovered(held, ai_held, a, n, rng):.3f}")
        tqdm.write(
            f"{conf}: vocab {len(rates[1])}, human train {len(h_train)}, AI train {len(q_train)}, recovery {' '.join(rec)}"
        )
        for year, g in R.groupby("year"):
            g = g[~g.paper_id.isin(train)]
            rows.append(dict(conf=conf, year=year, n=len(g), **estimate(rates, g.text, boot_rng)))
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
            self.bert = AutoModel.from_pretrained(COCODET_BASE)
            h = self.bert.config.hidden_size
            self.main_classifier_weight = nn.Parameter(torch.empty(3, h))
            self.content_classifier = nn.Linear(h, 8)
            self.style_classifier = nn.Linear(h, 7)
            self.fine_grained_classifier = nn.Linear(h, 6)

        def forward(self, input_ids, attention_mask):
            pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            cos = F.linear(F.normalize(pooled, dim=1), F.normalize(self.main_classifier_weight, dim=1))
            return (cos * 30.0).softmax(-1)

    model = CoCoDet()
    model.load_state_dict(torch.load(hf_hub_download("khaaaaaan/CoCoDet", "CoCoDet.bin"), map_location="cpu"))
    return model.eval()


def cocodet_order(df):
    df = df.assign(text=df.text.str.replace(TAGS + r"\s*", "", regex=True)).sample(frac=1, random_state=0)
    df["chunk"] = range(len(df))
    df["chunk"] //= 256
    return df.assign(n=df.text.str.len()).sort_values(["chunk", "n"])


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

    tok = AutoTokenizer.from_pretrained(COCODET_BASE)
    model = load_cocodet()
    with open(out, "a") as f, torch.inference_mode():
        for k, i in enumerate(tqdm(range(0, len(df), batch), desc="cocodet batches")):
            b = df.iloc[i : i + batch]
            enc = tok(b.text.tolist(), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            probs = model(enc["input_ids"], enc["attention_mask"]).tolist()
            f.writelines(
                json.dumps(dict(review_id=rid, **dict(zip(COCODET_CLASSES, p)))) + "\n" for rid, p in zip(b.review_id, probs)
            )
            if k % 250 == 0:
                f.flush()


def style_features(texts):
    t = clean(pd.Series(texts))
    return hstack([WORD_HASH.transform(t), CHAR_HASH.transform(t)]).tocsr()


def p_llm(clf, tfidf, texts, desc):
    starts = tqdm(range(0, len(texts), 5000), desc=desc)
    return np.concatenate([clf.predict_proba(tfidf.transform(style_features(texts.iloc[i : i + 5000])))[:, 1] for i in starts])


def style(out=settings.DATA_DIR / "detector_scores.parquet", by_cell=settings.REPORTS_DIR / "detector_by_cell.csv"):
    intel = pd.read_parquet(settings.DATA_DIR / "intel_reviews.parquet", columns=["label", "llm", "year", "text"])
    train, test = intel[intel.year.isin([2021, 2022, 2023])], intel[intel.year == 2024]
    del intel
    F_train = style_features(train.text)
    tfidf = TfidfTransformer(sublinear_tf=True).fit(F_train)
    clf = LogisticRegression(C=4.0, max_iter=3000).fit(tfidf.transform(F_train), train.label)
    del F_train
    p_te = clf.predict_proba(tfidf.transform(style_features(test.text)))[:, 1]
    print(f"train {len(train)}, test 2024 pairs {len(test)}: AUC {roc_auc_score(test.label, p_te):.4f}")
    for m, g in test.assign(p=p_te).groupby("llm"):
        print(f"  {m:8s} AUC {roc_auc_score(g.label, g.p):.4f}  TPR@p>0.5 {(g[g.label == 1].p > 0.5).mean():.3f}")

    S = pd.read_parquet(settings.DATA_DIR / "sample_reviews.parquet", columns=["review_id", "conf", "year", "text"])
    S["p_llm"] = p_llm(clf, tfidf, S.text, "score reviews")
    pre = S[(S.conf == "ICLR") & S.year.between(2017, 2022)].p_llm
    q90, q99 = np.quantile(pre, 0.90), np.quantile(pre, 0.99)
    S["llm_flag"] = S.p_llm >= q99
    S["llm_grade"] = np.select([S.p_llm < q90, S.p_llm < q99, S.p_llm <= 0.95], [0, 1, 2], 3)
    print(f"pre-period thresholds: q90={q90:.3f}, q99={q99:.3f}")
    print("share flagged (>= pre-period 99th pct) by venue-year:")
    print(
        S.groupby(["conf", "year"])
        .agg(
            flag=("llm_flag", "mean"),
            p_mean=("p_llm", "mean"),
            g2plus=("llm_grade", lambda g: (g >= 2).mean()),
            g3=("llm_grade", lambda g: (g == 3).mean()),
        )
        .round(3)
        .to_string()
    )

    ss = S.text.str.extractall(r"\[(SUMMARY|STRENGTHS)\](.*?)(?=\[[A-Z]+\]|\Z)")[1].groupby(level=0).apply(" ".join)
    S["p_llm_ss"] = np.nan
    has = ss.index[ss.str.len() > 200]
    S.loc[has, "p_llm_ss"] = p_llm(clf, tfidf, ss.loc[has], "score summary+strengths")
    S["llm_flag_ss"] = S.p_llm_ss >= q99
    print("summary+strengths exposure available for", len(has), "reviews; flagged share by year (ICLR 2024+):")
    print(S[(S.conf == "ICLR") & (S.year >= 2024)].groupby("year").llm_flag_ss.mean().round(3).to_dict())
    S[["review_id", "p_llm", "llm_flag", "llm_grade", "p_llm_ss", "llm_flag_ss"]].to_parquet(out)

    gz = pd.read_parquet(settings.DATA_DIR / "gptzero.parquet", columns=["review_id", "gz_ai", "gz_isAI"])
    m = S.merge(gz, on="review_id")
    for y in (2023, 2024):
        g = m[m.year == y]
        print(
            f"ICLR {y}: GPTZero isAI {g.gz_isAI.mean():.3f}, ours flagged {g.llm_flag.mean():.3f}; "
            f"AUC of p_llm for GPTZero isAI {roc_auc_score(g.gz_isAI, g.p_llm):.3f}; "
            f"corr(p_llm, gz_ai) {np.corrcoef(g.p_llm, g.gz_ai)[0, 1]:.3f}; "
            f"P(flag|isAI) {g[g.gz_isAI].llm_flag.mean():.3f}, P(isAI|flag) {g[g.llm_flag].gz_isAI.mean():.3f}"
        )
    S.groupby(["conf", "year"]).agg(
        flag=("llm_flag", "mean"), p_mean=("p_llm", "mean"), g2plus=("llm_grade", lambda g: (g >= 2).mean())
    ).round(4).to_csv(by_cell)


if __name__ == "__main__":
    fire.Fire(
        {"reference": reference, "wordshare": wordshare, "wordshare_fields": wordshare_fields, "cocodet": cocodet, "style": style}
    )
