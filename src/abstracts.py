from pathlib import Path

import fire
import pandas as pd
import pyarrow.dataset as ds

import hub
from settings import settings


def build_input(out=settings.RAW_DIR / "abstract_input.parquet"):
    papers = hub.frame("papers", ["conf", "year", "paper_id", "abstract"], ds.field("abstract").is_valid())
    papers = papers[papers.abstract.str.strip().ne("")]
    papers = papers.assign(key=papers.conf + ":" + papers.paper_id, set="abstract").rename(columns={"abstract": "text"})
    assert papers.key.is_unique
    print(papers.groupby("conf").size().to_string())
    papers.to_parquet(out, index=False)


def flags(
    input=settings.RAW_DIR / "abstract_input.parquet",
    out=settings.RAW_DIR / "abstract_flags.parquet",
    reports=settings.REPORTS_DIR,
):
    df = pd.read_parquet(input, columns=["key", "conf", "year", "paper_id"])
    summary = {}
    for name in settings.ABSTRACT_DETECTORS:
        scored = pd.read_json(settings.RAW_DIR / f"{name}_abstract_scores.jsonl", lines=True)[["key", "score", "n_tokens"]]
        score, tokens, flag = (f"abstract_{name}_{kind}" for kind in ("score", "tokens", "flag"))
        df = df.merge(scored.rename(columns={"score": score, "n_tokens": tokens}), on="key")
        calibration = df[df.year <= settings.MELD_CALIBRATION_LAST_YEAR][score]
        threshold = calibration.quantile(settings.MELD_FLAG_QUANTILE)
        df[flag] = df[score] > threshold
        summary |= {f"{name}_threshold": threshold, f"{name}_calibration_abstracts": len(calibration)}
    rates = df.groupby(["conf", "year"])[[f"abstract_{name}_flag" for name in settings.ABSTRACT_DETECTORS]].mean().mul(100)
    rates.insert(0, "n", df.groupby(["conf", "year"]).size())
    rates.round(3).to_csv(Path(reports) / "abstract_rates.csv")
    print(rates.round(2).to_string())
    pd.Series(summary).to_csv(Path(reports) / "abstract_summary.csv", header=["value"], index_label="name")
    print(pd.Series(summary).round(3).to_string())
    df.drop(columns=["key"]).to_parquet(out, index=False)


if __name__ == "__main__":
    fire.Fire({"input": build_input, "flags": flags})
