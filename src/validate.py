import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from outcomes import ICLR_FROM, SOURCE_FILES, in_window
from settings import settings

BATCH_ROWS = 20000


def se(x):
    return x.std(ddof=1) / np.sqrt(x.count())


def robust(model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR, tex=settings.TABLES_DIR / "robust.tex"):
    main = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet").query("conf == 'ICLR' and n_claims > 0")
    pair_table = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "hit_spec"]
    )
    specific = pair_table[pair_table.conf == "ICLR"].groupby(["panel_id", "year"]).hit_spec.mean().reset_index()
    sub3 = pd.read_parquet(settings.ANALYSIS_DIR / f"sub3_outcomes_{model}.parquet").query("conf == 'ICLR' and n_claims > 0")
    noq = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}_noq.parquet").query("n_claims > 0")
    wd = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}_wd.parquet").query("n_claims > 0")
    with_wd = pd.concat([main, wd.assign(year=2026)])
    checks = [
        ("Main overlap rate", main, "hit_rate_strict", 2023),
        ("Specific claims only", specific, "hit_spec", 2023),
        ("Same part of paper (lenient)", main, "hit_rate_lenient", 2023),
        ("3-review subsets, grouped again", sub3, "hit_rate_strict", 2023),
        ("Questions field removed", noq, "hit_rate_strict", 2024),
        ("Withdrawn 2026 papers added", with_wd, "hit_rate_strict", 2023),
    ]
    rows = []
    for name, df, col, base_year in checks:
        base = df[df.year == base_year][col].dropna() * 100
        last = df[df.year == 2026][col].dropna() * 100
        rows.append(
            {
                "check": name,
                "base_year": base_year,
                "base": base.mean(),
                "y2026": last.mean(),
                "change": last.mean() - base.mean(),
                "se": np.hypot(se(base), se(last)),
            }
        )
    out = pd.DataFrame(rows)
    out.round(3).to_csv(Path(reports) / f"robust_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Check & Base year & Base & 2026 & Change (SE) \\", r"\midrule"]
    lines += [
        f"{r.check} & {r.base_year} & {r.base:.1f} & {r.y2026:.1f} & {r.change:+.1f} ({r.se:.1f}) \\\\" for r in out.itertuples()
    ]
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def fe_gap(design, flag, controls=()):
    import statsmodels.api as sm

    n_flagged = flag.reindex(design.a).to_numpy() + flag.reindex(design.b).to_numpy()
    X = pd.DataFrame(
        {"one": (n_flagged == 1) * 1.0, "two": (n_flagged == 2) * 1.0, "hit": design.hit.to_numpy()}, index=design.index
    )
    X = X.join(design[list(controls)])
    X = X - X.groupby(design.panel_id.to_numpy()).transform("mean")
    fit = sm.OLS(X.hit, X[["one", "two", *controls]]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(design.panel_id)[0]}
    )
    return {
        "one": fit.params["one"] * 100,
        "one_se": fit.bse["one"] * 100,
        "two": fit.params["two"] * 100,
        "two_se": fit.bse["two"] * 100,
    }


def detector_placebo(model=settings.DEFAULT_MODEL, share=0.45, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    data, reports = Path(data), Path(reports)
    iclr_pairs = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    iclr_pairs = iclr_pairs[iclr_pairs.conf == "ICLR"]
    scores = pd.read_parquet(data / "detector_scores.parquet", columns=["review_id", "p_llm", "llm_flag"]).set_index("review_id")
    trained = set(pd.read_parquet(data / "intel_reviews.parquet", columns=["paper_id", "year"]).query("year <= 2023").paper_id)
    periods = {
        str(ICLR_FROM): (ICLR_FROM, ICLR_FROM),
        "2021-2022": (2021, 2022),
        "2023": (2023, 2023),
        "2024-2026 (real flags)": (2024, 2026),
    }
    both_scored = iclr_pairs.a.isin(scores.index) & iclr_pairs.b.isin(scores.index)
    rows = []
    for name, (lo, hi) in tqdm(periods.items(), desc="periods"):
        period_pairs = iclr_pairs[iclr_pairs.year.between(lo, hi) & both_scored]
        has_real_flags = lo >= 2024
        if has_real_flags:
            flag = scores.llm_flag.astype(int)
        else:
            reviews = pd.Index(np.union1d(period_pairs.a, period_pairs.b))
            score = scores.p_llm.reindex(reviews)
            flag = (score >= score.quantile(1 - share)).astype(int)
        untrained = period_pairs[~period_pairs.panel_id.isin(trained)]
        for subset, subset_pairs in [("all", period_pairs), ("without detector-training papers", untrained)]:
            rows.append(dict(period=name, subset=subset, papers=subset_pairs.panel_id.nunique(), **fe_gap(subset_pairs, flag)))
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"detector_placebo_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def short_long_logs(values, design):
    log_a = np.log(values.reindex(design.a).to_numpy())
    log_b = np.log(values.reindex(design.b).to_numpy())
    return np.minimum(log_a, log_b), np.maximum(log_a, log_b)


def batches(path, columns, ids, desc):
    parquet = pq.ParquetFile(path)
    wanted = pa.array(list(ids), type=parquet.schema_arrow.field("review_id").type)
    n_batches = -(-parquet.metadata.num_rows // BATCH_ROWS)
    for batch in tqdm(parquet.iter_batches(batch_size=BATCH_ROWS, columns=columns), total=n_batches, desc=desc):
        yield batch.filter(pc.is_in(batch.column("review_id"), value_set=wanted))


def review_chars(data, ids):
    parts = []
    for name in ["reviews.parquet", *SOURCE_FILES.values()]:
        for batch in batches(data / name, ["review_id", "text"], ids, name):
            lengths = pc.utf8_length(batch.column("text")).to_numpy()
            parts.append(pd.Series(lengths, index=batch.column("review_id").to_pandas()))
    chars = pd.concat(parts)
    return chars[~chars.index.duplicated()]


def pair_design(model, data):
    design = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    chars = review_chars(data, set(design.a) | set(design.b))
    design["log_short"], design["log_long"] = short_long_logs(chars, design)
    return design


def rq1_evidence(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    data, reports = Path(data), Path(reports)
    flag = pd.read_parquet(data / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag
    flag = flag.astype(int)
    design = pair_design(model, data)
    claim_counts = pd.read_parquet(data / f"claims_table_{model}.parquet", columns=["review_id"]).groupby("review_id").size()
    design["claims_short"], design["claims_long"] = short_long_logs(claim_counts, design)
    iclr = design[(design.conf == "ICLR") & (design.year >= 2024)].dropna()
    coco_raw = pd.concat([pd.read_json(f, lines=True) for f in [data / "cocodet_main.jsonl", data / "cocodet_icml.jsonl"]])
    coco_flag = (coco_raw.set_index("review_id")[["human", "mix", "ai"]].idxmax(axis=1) != "human").astype(int)
    coco_flag = coco_flag[~coco_flag.index.duplicated()]
    both_coco = iclr[iclr.a.isin(coco_flag.index) & iclr.b.isin(coco_flag.index)]
    rows = []

    def add(group, label, est, err, papers):
        rows.append(dict(group=group, label=label, est=est, se=err, papers=papers))

    pipe = "Pipeline"
    for label, subset_pairs, pair_flags, controls in [
        ("ICLR 2024–26", iclr, flag, ()),
        ("+ length and claim count", iclr, flag, ("log_short", "log_long", "claims_short", "claims_long")),
        ("NeurIPS 2024–25", design[(design.conf == "NeurIPS") & (design.year >= 2024)], flag, ()),
        ("CoCoDet labels, ICLR", both_coco, coco_flag, ()),
    ]:
        gap = fe_gap(subset_pairs, pair_flags, controls)
        add(pipe, label, gap["two"], gap["two_se"], subset_pairs.panel_id.nunique())
    placebo = pd.read_csv(reports / f"detector_placebo_{model}.csv").query("subset == 'all' and not period.str.contains('real')")
    for r in placebo.itertuples():
        add("Placebo before ChatGPT", f"Top-45% scores, ICLR {r.period}", r.two, r.two_se, r.papers)
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"rq1_evidence_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    fire.Fire(
        {
            "robust": robust,
            "detector_placebo": detector_placebo,
            "rq1_evidence": rq1_evidence,
        }
    )
