import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from settings import settings

COVER_MIN_SIM = 0.5
VARIANT_COLS = [
    "S3",
    "D3",
    "U3",
    "share_shared_claims",
    "hit_rate_strict",
    "share_specific",
    "claims_per_review",
]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_flags(data):
    return (
        pd.read_parquet(Path(data) / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag
    )


def clustered(model, *groups):
    codes = [pd.factorize(g)[0] for g in groups]
    return model.fit(cov_type="cluster", cov_kwds={"groups": codes[0] if len(codes) == 1 else np.column_stack(codes)})


def se(x):
    return x.std(ddof=1) / np.sqrt(x.count())


def ampere(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    from scipy.stats import spearmanr

    from between import openai_embed

    data = Path(data)
    reviews = pd.read_parquet(data / "ampere_full.parquet").set_index("review_id")
    claims = {
        r["review_id"]: [c["claim"] for c in r["claims"]]
        for r in read_jsonl(data / f"claims_{model}_ampere.jsonl")
        if "claims" in r
    }
    reviews = reviews[reviews.index.isin(claims)]
    reviews["n_claims"] = reviews.index.map(lambda i: len(claims[i]))
    print(len(reviews), "reviews")
    for k in ["n_prop", "n_eval", "n_req"]:
        print(f"spearman(n_claims, {k}) = {spearmanr(reviews.n_claims, reviews[k]).statistic:.3f}")
    reviews["n_crit"] = reviews.n_eval + reviews.n_req
    print(
        f"spearman(n_claims, n_eval+n_req) = {spearmanr(reviews.n_claims, reviews.n_crit).statistic:.3f}; "
        f"means claims {reviews.n_claims.mean():.1f}, eval+req {reviews.n_crit.mean():.1f}, req {reviews.n_req.mean():.1f}"
    )

    covered_req, covered_eval, claim_on_argument = [], [], []
    for rid, row in tqdm(reviews.iterrows(), total=len(reviews), desc="ampere reviews"):
        review_claims = claims[rid]
        if not review_claims:
            covered_req += [False] * row.n_req
            covered_eval += [False] * row.n_eval
            continue
        sents = list(row.sents)
        emb = openai_embed(sents + review_claims, settings.EMBED_MODEL, batch=512, threads=1)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        sim = emb[: len(sents)] @ emb[len(sents) :].T
        for best, label in zip(sim.max(1), row.labels):
            if label == "request":
                covered_req.append(best >= COVER_MIN_SIM)
            if label == "evaluation":
                covered_eval.append(best >= COVER_MIN_SIM)
        is_argument = [label in ("evaluation", "request") for label in row.labels]
        claim_on_argument += [is_argument[nearest] for nearest in sim.argmax(0)]
    print(f"request propositions covered by a claim: {np.mean(covered_req):.3f} (n={len(covered_req)})")
    print(f"evaluation propositions covered by a claim: {np.mean(covered_eval):.3f} (n={len(covered_eval)})")
    print(
        f"claims whose nearest proposition is evaluation/request: {np.mean(claim_on_argument):.3f} (n={len(claim_on_argument)})"
    )


def nonempty_outcomes(path):
    outcomes = pd.read_parquet(path)
    return outcomes[outcomes.n_claims > 0]


def variants(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    data = Path(data)
    full = nonempty_outcomes(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet").set_index("panel_id")
    out = {}
    sub3 = nonempty_outcomes(settings.ANALYSIS_DIR / f"sub3_outcomes_{model}.parquet").set_index("paper_id")
    sub3_joined = sub3.join(full[["S3", "D3", "share_shared_claims", "hit_rate_strict"]], suffixes=("", "_full"))
    sub3_joined = sub3_joined.dropna(subset=["S3", "S3_full"])
    out["sub3"] = pd.DataFrame(
        [
            {
                "year": y,
                "n": len(g),
                "S3_recluster": g.S3.mean(),
                "S3_rarefied": g.S3_full.mean(),
                "D3_recluster": g.D3.mean(),
                "D3_rarefied": g.D3_full.mean(),
                "share_recluster": g.share_shared_claims.mean(),
                "share_rarefied": g.share_shared_claims_full.mean(),
                "se_S3": se(g.S3),
            }
            for y, g in sub3_joined.groupby("year")
        ]
    ).set_index("year")
    noq = nonempty_outcomes(settings.ANALYSIS_DIR / f"panel_outcomes_{model}_noq.parquet").set_index("paper_id")
    noq_joined = noq.join(full[VARIANT_COLS], suffixes=("", "_full")).dropna(subset=["S3", "S3_full"])
    noq_cols = ["S3", "share_shared_claims", "claims_per_review", "D3"]
    out["noq"] = pd.DataFrame(
        [
            dict(
                year=y,
                n=len(g),
                **{f"{c}_noq": g[c].mean() for c in noq_cols},
                **{f"{c}_full": g[f"{c}_full"].mean() for c in noq_cols},
                se_S3=se(g.S3),
            )
            for y, g in noq_joined.groupby("year")
        ]
    ).set_index("year")
    withdrawn = nonempty_outcomes(settings.ANALYSIS_DIR / f"panel_outcomes_{model}_wd.parquet")
    kept = full[(full.conf == "ICLR") & (full.year == 2026)]
    out["wd"] = pd.DataFrame(
        {
            "withdrawn": withdrawn[VARIANT_COLS].mean(),
            "withdrawn_se": withdrawn[VARIANT_COLS].agg(se),
            "kept": kept[VARIANT_COLS].mean(),
            "kept_se": kept[VARIANT_COLS].agg(se),
        }
    )
    out["wd"].loc["n"] = [len(withdrawn), np.nan, len(kept), np.nan]
    for name, table in out.items():
        table.round(4).to_csv(Path(reports) / f"variant_{name}_{model}.csv")
        print(f"== {name}\n{table.round(3).to_string()}\n")


def robust(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "robust.tex",
):
    data = Path(data)
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
    fit = clustered(sm.OLS(X.hit, X[["one", "two", *controls]]), design.panel_id)
    return {
        "one": fit.params["one"] * 100,
        "one_se": fit.bse["one"] * 100,
        "two": fit.params["two"] * 100,
        "two_se": fit.bse["two"] * 100,
    }


def detector_placebo(model=settings.DEFAULT_MODEL, share=0.45, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    from outcomes import ICLR_FROM

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


def rq1_evidence(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    from outcomes import pair_design

    data, reports = Path(data), Path(reports)
    flag = load_flags(data).astype(int)
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


def f1000_reviewer_names(xml_dir):
    import xml.etree.ElementTree as ET

    names = {}
    for f in tqdm(sorted(Path(xml_dir).glob("*.xml")), desc="f1000 xml"):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        for sub in root.iter("sub-article"):
            meta = sub.find("front-stub")
            if meta is None:
                meta = sub.find("front/article-meta")
            if meta is None:
                continue
            person = meta.find(".//contrib/name")
            if person is not None:
                full_name = f"{person.findtext('given-names')} {person.findtext('surname')}"
                names[meta.findtext("article-id[@pub-id-type='doi']")] = full_name.lower()
    return pd.Series(names)


def reviewer_fe(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    import statsmodels.formula.api as smf

    data, reports = Path(data), Path(reports)
    reviewer_name = f1000_reviewer_names(settings.RAW_JOURNAL_REVIEWS_DIR / "f1000")
    f1000 = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    f1000 = f1000[(f1000.conf == "F1000") & (f1000.year >= 2019) & (f1000.year != 2024)]
    sides = pd.concat([f1000.rename(columns={"a": "rid", "b": "other"}), f1000.rename(columns={"b": "rid", "a": "other"})])
    per_review = sides.groupby(["rid", "panel_id", "year"], as_index=False).agg(hit=("hit", "mean"))
    per_review["name"] = per_review.rid.map(reviewer_name)
    per_review["post"] = (per_review.year >= 2025).astype(int)
    per_review["hit"] *= 100
    span = per_review.groupby("name").post.agg(["min", "max"])
    seen_both = per_review[per_review.name.isin(span.index[(span["min"] == 0) & (span["max"] == 1)])]

    def fit(label, formula, df):
        result = clustered(smf.ols(formula, data=df), df.panel_id, df.name.fillna(df.rid))
        return dict(
            label=label,
            est=result.params["post"],
            se=result.bse["post"],
            reports=int(result.nobs),
            reviewers=df.name.nunique(),
            pre=df[df.post == 0].hit.mean(),
            post=df[df.post == 1].hit.mean(),
        )

    out = pd.DataFrame(
        [
            fit("All reports, pooled", "hit ~ post", per_review),
            fit("Reviewers seen in both periods, pooled", "hit ~ post", seen_both),
            fit("Reviewers seen in both periods, reviewer fixed effects", "hit ~ post + C(name)", seen_both),
        ]
    )
    out.to_csv(reports / f"reviewer_fe_f1000_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def paper_side(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    import statsmodels.formula.api as smf

    from outcomes import pair_design

    data, reports = Path(data), Path(reports)
    abstract_labels = (
        pd.read_json(data / "cocodet_abstracts.jsonl", lines=True).drop_duplicates("review_id").set_index("review_id")
    )
    score = abstract_labels["ai"] + abstract_labels["mix"]
    design = pair_design(model, data)
    design = design[design.conf == "ICLR"]
    design["score"] = score.reindex(design.panel_id).to_numpy()
    design = design.dropna(subset=["score"])
    paper_scores = design.drop_duplicates("panel_id").score
    design["z"] = (design.score - paper_scores.mean()) / paper_scores.std()
    flag = load_flags(data)
    design["nflag"] = flag.reindex(design.a).to_numpy().astype(float) + flag.reindex(design.b).to_numpy().astype(float)
    papers = design.groupby("year").panel_id.nunique()
    print(f"{design.panel_id.nunique()} ICLR papers with an abstract score; by year {papers.to_dict()}")
    abstracts = design.drop_duplicates("panel_id")
    print(
        "share of abstracts CoCoDet labels AI or mixed:",
        abstracts.groupby("year").score.apply(lambda s: (s > 0.5).mean()).round(3).to_dict(),
    )
    neither_flagged = design[design.nflag == 0]
    both_flagged = design[design.nflag == 2]
    rows = []
    for subset, subset_pairs in [("all pairs", design), ("neither flagged", neither_flagged), ("both flagged", both_flagged)]:
        paper_rates = subset_pairs.groupby(["panel_id", "year"], as_index=False).agg(hit=("hit", "mean"), z=("z", "first"))
        fit = clustered(smf.ols("hit ~ C(year) + z", data=paper_rates), paper_rates.panel_id)
        rows.append(dict(subset=subset, est=fit.params["z"] * 100, se=fit.bse["z"] * 100, papers=len(paper_rates)))
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"paper_side_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    fire.Fire(
        {
            "ampere": ampere,
            "variants": variants,
            "robust": robust,
            "detector_placebo": detector_placebo,
            "rq1_evidence": rq1_evidence,
            "reviewer_fe": reviewer_fe,
            "paper_side": paper_side,
        }
    )
