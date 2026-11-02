import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from settings import settings

TAGS = r"\[(SUMMARY|MAIN|STRENGTHS|WEAKNESSES|QUESTIONS|LIMITATIONS|OTHER)\]"
SUMMARY_STRENGTHS = r"\[(SUMMARY|STRENGTHS)\](.*?)(?=\[[A-Z]+\]|\Z)"
HASH_FEATURES = 2**18
TRAIN_YEARS = [2021, 2022, 2023]
TEST_YEAR = 2024
CALIBRATION_YEARS = (2017, 2022)
SCORE_CHUNK = 5000
GRADE3_CUTOFF = 0.95
SUMMARY_STRENGTHS_MIN_CHARS = 200
WORD = HashingVectorizer(
    ngram_range=(1, 2), n_features=HASH_FEATURES, alternate_sign=False, norm=None, lowercase=True, dtype=np.float32
)
CHAR = HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 4),
    n_features=HASH_FEATURES,
    alternate_sign=False,
    norm=None,
    lowercase=True,
    dtype=np.float32,
)


def features(texts):
    cleaned = pd.Series(texts).str.replace(TAGS, " ", regex=True)
    return hstack([WORD.transform(cleaned), CHAR.transform(cleaned)]).tocsr()


def p_llm(clf, tfidf, texts):
    chunks = [texts.iloc[i : i + SCORE_CHUNK] for i in range(0, len(texts), SCORE_CHUNK)]
    return np.concatenate([clf.predict_proba(tfidf.transform(features(chunk)))[:, 1] for chunk in chunks])


def scores(out, data=str(settings.DATA_DIR)):
    data = Path(data)
    intel = pd.read_parquet(data / "intel_reviews.parquet")
    train = intel[intel.year.isin(TRAIN_YEARS)]
    test = intel[intel.year == TEST_YEAR]
    del intel
    tfidf = TfidfTransformer(sublinear_tf=True).fit(features(train.text))
    train_x = tfidf.transform(features(train.text))
    test_x = tfidf.transform(features(test.text))
    print("features built", train_x.shape, flush=True)
    clf = LogisticRegression(C=4.0, max_iter=3000).fit(train_x, train.label)
    p_test = clf.predict_proba(test_x)[:, 1]
    print(f"train {len(train)}, test {TEST_YEAR} pairs {len(test)}: AUC {roc_auc_score(test.label, p_test):.4f}")
    for llm, g in test.assign(p=p_test).groupby("llm"):
        print(f"  {llm:8s} AUC {roc_auc_score(g.label, g.p):.4f}  TPR@p>0.5 {(g[g.label == 1].p > 0.5).mean():.3f}")

    scored = pd.read_parquet(data / "sample_reviews.parquet", columns=["review_id", "conf", "year", "text"])
    scored["p_llm"] = p_llm(clf, tfidf, scored.text)
    pre = scored[(scored.conf == "ICLR") & scored.year.between(*CALIBRATION_YEARS)].p_llm
    q90, q99 = np.quantile(pre, 0.90), np.quantile(pre, 0.99)
    scored["llm_flag"] = scored.p_llm >= q99
    scored["llm_grade"] = np.select([scored.p_llm < q90, scored.p_llm < q99, scored.p_llm <= GRADE3_CUTOFF], [0, 1, 2], 3)
    print(f"pre-period thresholds: q90={q90:.3f}, q99={q99:.3f}")
    by_venue_year = scored.groupby(["conf", "year"]).agg(
        flag=("llm_flag", "mean"),
        p_mean=("p_llm", "mean"),
        g2plus=("llm_grade", lambda g: (g >= 2).mean()),
        g3=("llm_grade", lambda g: (g == 3).mean()),
    )
    print(by_venue_year.round(3).to_string())

    summary_strengths = scored.text.str.extractall(SUMMARY_STRENGTHS)[1].groupby(level=0).apply(" ".join)
    scored["p_llm_ss"] = np.nan
    has = summary_strengths.index[summary_strengths.str.len() > SUMMARY_STRENGTHS_MIN_CHARS]
    scored.loc[has, "p_llm_ss"] = p_llm(clf, tfidf, summary_strengths.loc[has])
    scored["llm_flag_ss"] = scored.p_llm_ss >= q99
    print("summary+strengths scored for", len(has), "reviews")
    scored[["review_id", "p_llm", "llm_flag", "llm_grade", "p_llm_ss", "llm_flag_ss"]].to_parquet(out)


if __name__ == "__main__":
    fire.Fire({"scores": scores})
