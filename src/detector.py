"""Style-based LLM-text detector for reviews, trained on the IntelLabs paired human/LLM reviews (ICLR 2021-2023),
tested on the 2024 pairs, calibrated on our pre-ChatGPT reviews (ICLR 2017-2022), and compared with GPTZero on ICLR 2024.

It uses only surface features (hashed word 1-2-grams and character 3-5-grams), never similarity to an LLM review of the
same paper, so it is not circular with claim overlap. Output: data/detector_scores.parquet with p_llm, llm_flag
(above the 99th percentile of pre-ChatGPT reviews), llm_grade (0: below pre-period 90th percentile; 1: 90-99th;
2: above 99th; 3: p_llm > 0.95).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
word = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 18, alternate_sign=False, norm=None, lowercase=True, dtype=np.float32)
char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=2 ** 18, alternate_sign=False, norm=None, lowercase=True, dtype=np.float32)


def strip_tags(s):
    return s.str.replace(r"\[(SUMMARY|MAIN|STRENGTHS|WEAKNESSES|QUESTIONS|LIMITATIONS|OTHER)\]", " ", regex=True)


def feats(texts):
    t = strip_tags(pd.Series(texts))
    return hstack([word.transform(t), char.transform(t)]).tocsr()


I = pd.read_parquet(DATA / "intel_reviews.parquet")
train = I[I.year.isin([2021, 2022, 2023])]
test = I[I.year == 2024]
del I
tfidf = TfidfTransformer(sublinear_tf=True).fit(feats(train.text))
Xtr = tfidf.transform(feats(train.text))
Xte = tfidf.transform(feats(test.text))
print("features built", Xtr.shape, flush=True)
clf = LogisticRegression(C=4.0, max_iter=3000).fit(Xtr, train.label)
p_te = clf.predict_proba(Xte)[:, 1]
print(f"train {len(train)}, test 2024 pairs {len(test)}: AUC {roc_auc_score(test.label, p_te):.4f}")
for m, g in test.assign(p=p_te).groupby("llm"):
    print(f"  {m:8s} AUC {roc_auc_score(g.label, g.p):.4f}  TPR@p>0.5 {(g[g.label == 1].p > 0.5).mean():.3f}")

S = pd.read_parquet(DATA / "sample_reviews.parquet", columns=["review_id", "conf", "year", "text"])
S["p_llm"] = np.concatenate([clf.predict_proba(tfidf.transform(feats(S.text.iloc[i:i + 5000])))[:, 1] for i in range(0, len(S), 5000)])
pre = S[(S.conf == "ICLR") & S.year.between(2017, 2022)].p_llm
q90, q99 = np.quantile(pre, 0.90), np.quantile(pre, 0.99)
S["llm_flag"] = S.p_llm >= q99
S["llm_grade"] = np.select([S.p_llm < q90, S.p_llm < q99, S.p_llm <= 0.95], [0, 1, 2], 3)
print(f"pre-period thresholds: q90={q90:.3f}, q99={q99:.3f}")
print("share flagged (>= pre-period 99th pct) by venue-year:")
print(S.groupby(["conf", "year"]).agg(flag=("llm_flag", "mean"), p_mean=("p_llm", "mean"), g2plus=("llm_grade", lambda g: (g >= 2).mean()), g3=("llm_grade", lambda g: (g == 3).mean())).round(3).to_string())

# exposure measured on fields that feed no claims (summary + strengths), for the 2024+ structured forms
ss = S.text.str.extractall(r"\[(SUMMARY|STRENGTHS)\](.*?)(?=\[[A-Z]+\]|\Z)")[1].groupby(level=0).apply(" ".join)
S["p_llm_ss"] = np.nan
has = ss.index[ss.str.len() > 200]
S.loc[has, "p_llm_ss"] = np.concatenate([clf.predict_proba(tfidf.transform(feats(ss.loc[has].iloc[i:i + 5000])))[:, 1] for i in range(0, len(has), 5000)])
S["llm_flag_ss"] = S.p_llm_ss >= q99
print("summary+strengths exposure available for", len(has), "reviews; flagged share by year (ICLR 2024+):")
print(S[(S.conf == "ICLR") & (S.year >= 2024)].groupby("year").llm_flag_ss.mean().round(3).to_dict())
S[["review_id", "p_llm", "llm_flag", "llm_grade", "p_llm_ss", "llm_flag_ss"]].to_parquet(DATA / "detector_scores.parquet")
gz = pd.read_parquet(DATA / "gptzero.parquet", columns=["review_id", "gz_ai", "gz_isAI"])
m = S.merge(gz, on="review_id")
for y in (2023, 2024):
    g = m[m.year == y]
    print(f"ICLR {y}: GPTZero isAI {g.gz_isAI.mean():.3f}, ours flagged {g.llm_flag.mean():.3f}; AUC of p_llm for GPTZero isAI {roc_auc_score(g.gz_isAI, g.p_llm):.3f}; "
          f"corr(p_llm, gz_ai) {np.corrcoef(g.p_llm, g.gz_ai)[0, 1]:.3f}; P(flag|isAI) {g[g.gz_isAI].llm_flag.mean():.3f}, P(isAI|flag) {g[g.llm_flag].gz_isAI.mean():.3f}")
S.groupby(["conf", "year"]).agg(flag=("llm_flag", "mean"), p_mean=("p_llm", "mean"), g2plus=("llm_grade", lambda g: (g >= 2).mean())).round(4).to_csv(REPORTS / "detector_by_cell.csv")
