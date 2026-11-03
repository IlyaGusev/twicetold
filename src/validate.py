import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm

from extract import read_jsonl
from metrics import assign, load_jsonl, mean_overlap, review_groups
from outcomes import ICLR_FROM, SOURCE_FILES, in_window
from settings import settings

BATCH_ROWS = 20000
ML_VENUES = ["ICLR", "NeurIPS", "ICML"]
REMOVED_FIELD_TAGS = ("_nocq", "_noq")
FORM_CATEGORIES = ("clarity/presentation", "novelty", "reproducibility/details")
FIELD_LABELS = {
    "summary": "summary",
    "summary of the paper": "summary",
    "review": "review",
    "main review": "main review",
    "strength and weaknesses": "strengths and weaknesses",
    "strengths": "strengths",
    "weaknesses": "weaknesses",
    "questions": "questions",
    "clarity, quality, novelty and reproducibility": r"\textbf{clarity, quality, novelty and reproducibility}",
    "summary of the review": "review summary",
}


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
                "se": np.hypot(base.sem(), last.sem()),
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


def form_panel(panel, claims, n_reviews):
    df = assign(panel, claims)
    kept = df[~df.category.isin(FORM_CATEGORIES)]
    groups = review_groups(df.review, df.cluster, n_reviews)
    shared = set().union(*(a & b for a, b in combinations(groups, 2)))
    on_form = set(df.cluster[df.category.isin(FORM_CATEGORIES)])
    return dict(
        hit=mean_overlap(groups),
        hit_without=mean_overlap(review_groups(kept.review, kept.cluster, n_reviews)),
        form_share=len(shared & on_form) / len(shared) if shared else np.nan,
    )


def forms(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "forms.tex",
):
    data = Path(data)
    main = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet").query(
        f"conf == 'ICLR' and year >= {ICLR_FROM} and n_claims > 0"
    )
    claims = {rid: row["claims"] for rid, row in load_jsonl(data / f"claims_{model}.jsonl", "review_id").items()}
    clusters = load_jsonl(data / f"clusters_{model}.jsonl", "panel_id")
    panels = pd.DataFrame(
        [
            dict(year=p.year, **form_panel(clusters[p.panel_id], claims, p.n_reviews))
            for p in tqdm(main.itertuples(), total=len(main))
        ],
        index=main.panel_id,
    )
    pair_table = pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "hit_spec"])
    panels["hit_spec"] = pair_table[pair_table.conf == "ICLR"].groupby("panel_id").hit_spec.mean().reindex(panels.index)
    removed = pd.concat(
        pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}{tag}.parquet").query("n_claims > 0")
        for tag in REMOVED_FIELD_TAGS
    )
    panels["hit_removed"] = removed.set_index("paper_id").hit_rate_strict.reindex(panels.index)
    panels["removed_change"] = panels.hit_removed - panels.hit
    reviews = pd.read_parquet(data / "reviews.parquet", columns=["conf", "year", "form"]).query("conf == 'ICLR'")
    form_of_year = reviews.groupby("year").form.agg(lambda f: f.mode()[0])
    cols = ["hit", "hit_removed", "hit_without", "hit_spec", "form_share"]
    by_year = panels.groupby("year")[cols + ["removed_change"]]
    out = pd.concat([by_year.mean(), by_year.sem().add_suffix("_se")], axis=1) * 100
    out.insert(0, "removed_papers", panels.groupby("year").hit_removed.count())
    out.insert(0, "papers", panels.groupby("year").size())
    out.insert(0, "form", form_of_year)
    out = out.reset_index()
    out.round(3).to_csv(Path(reports) / f"forms_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [
        r"\begin{tabular}{l>{\raggedright\arraybackslash}p{3.8cm}rrrrrr}",
        r"\toprule",
        r"Year & Text fields & Papers & Overlap & Field removed & No C/N/R & Specific & C/N/R shared \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        fields = sorted(r.form.split("+"), key=list(FIELD_LABELS).index)
        cells = ["--" if pd.isna(getattr(r, c)) else f"{getattr(r, c):.1f} ({getattr(r, c + '_se'):.1f})" for c in cols]
        lines.append(f"{r.year} & {'; '.join(FIELD_LABELS[f] for f in fields)} & {r.papers:,} & {' & '.join(cells)} \\\\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def fe_gap(design, flag, controls=()):
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


def meld_flags(data):
    return pd.read_parquet(
        Path(data) / "meld_flags.parquet", columns=["review_id", "year", "score", "flag", "written"]
    ).set_index("review_id")


def meld_flag_at(meld, quantile):
    threshold = meld[meld.year <= settings.MELD_CALIBRATION_LAST_YEAR].score.quantile(quantile)
    return (meld.score > threshold).astype(int)


def detector_placebo(model=settings.DEFAULT_MODEL, share=0.45, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    iclr_pairs = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    iclr_pairs = iclr_pairs[iclr_pairs.conf == "ICLR"]
    scores = meld_flags(data)
    periods = {
        str(ICLR_FROM): (ICLR_FROM, ICLR_FROM),
        "2021-2022": (2021, 2022),
        "2023": (2023, 2023),
    }
    rows = []
    for name, (lo, hi) in tqdm(periods.items(), desc="periods"):
        period_pairs = iclr_pairs[iclr_pairs.year.between(lo, hi)]
        reviews = pd.Index(np.union1d(period_pairs.a, period_pairs.b))
        score = scores.score.reindex(reviews)
        flag = (score >= score.quantile(1 - share)).astype(int)
        rows.append(dict(period=name, papers=period_pairs.panel_id.nunique(), **fe_gap(period_pairs, flag)))
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / f"detector_placebo_{model}.csv", index=False)
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
    meld = meld_flags(data)
    flag, written = meld.flag.astype(int), meld.written.astype(int)
    design = pair_design(model, data)
    claim_counts = pd.read_parquet(data / f"claims_table_{model}.parquet", columns=["review_id"]).groupby("review_id").size()
    design["claims_short"], design["claims_long"] = short_long_logs(claim_counts, design)
    iclr = design[(design.conf == "ICLR") & (design.year >= 2024)].dropna()
    rows = []

    def add(group, label, est, err, papers):
        rows.append(dict(group=group, label=label, est=est, se=err, papers=papers))

    pipe = "Pipeline"
    for label, subset_pairs, pair_flags, controls in [
        ("ICLR 2024–26", iclr, flag, ()),
        ("+ length and claim count", iclr, flag, ("log_short", "log_long", "claims_short", "claims_long")),
        ("NeurIPS 2024–25", design[(design.conf == "NeurIPS") & (design.year >= 2024)], flag, ()),
        ("ICML 2025–26", design[design.conf == "ICML"], flag, ()),
        ("AI-written, ICLR", iclr, written, ()),
        ("99th-percentile threshold, ICLR", iclr, meld_flag_at(meld, 0.99), ()),
        ("99.5th-percentile threshold, ICLR", iclr, meld_flag_at(meld, 0.995), ()),
    ]:
        gap = fe_gap(subset_pairs, pair_flags, controls)
        add(pipe, label, gap["two"], gap["two_se"], subset_pairs.panel_id.nunique())
    placebo = pd.read_csv(reports / f"detector_placebo_{model}.csv")
    for r in placebo.itertuples():
        add("Placebo before ChatGPT", f"Top-45% scores, ICLR {r.period}", r.two, r.two_se, r.papers)
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"rq1_evidence_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def journal_gaps(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR, first_year=2024):
    flag = meld_flags(data).flag.astype(int)
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    journals = pairs[~pairs.conf.isin(ML_VENUES) & (pairs.year >= first_year)]
    rows = []
    for source, source_pairs in [*journals.groupby("conf"), ("All journals", journals)]:
        n_flagged = source_pairs.a.map(flag) + source_pairs.b.map(flag)
        gap = fe_gap(source_pairs, flag)
        rows.append(
            dict(
                source=source,
                est=gap["two"],
                se=gap["two_se"],
                papers=source_pairs.panel_id.nunique(),
                both_papers=source_pairs[n_flagged == 2].panel_id.nunique(),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / f"journal_gaps_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def strict_overlap(review_ids, claims, clusters):
    group_of = {}
    for g, cluster in enumerate(clusters):
        for cid in cluster["claim_ids"]:
            group_of.setdefault(cid, g)
    groups = [{group_of.get(f"R{j}-{k}", f"R{j}-{k}") for k in range(len(claims[rid]))} for j, rid in enumerate(review_ids)]
    return mean_overlap(groups)


def matched_claims(clusters, n_ref, n_ours):
    valid = {f"R0-{k}" for k in range(n_ref)} | {f"R1-{k}" for k in range(n_ours)}
    hit = set()
    for cluster in clusters:
        ids = valid.intersection(cluster["claim_ids"])
        if {cid[:2] for cid in ids} == {"R0", "R1"}:
            hit |= ids
    return sum(cid.startswith("R0") for cid in hit), sum(cid.startswith("R1") for cid in hit)


def mean_interval(values, name):
    values = values.dropna()
    half = stats.t.ppf(0.975, len(values) - 1) * values.sem()
    return {
        name: values.mean(),
        f"{name}_se": values.sem(),
        f"{name}_low": values.mean() - half,
        f"{name}_high": values.mean() + half,
    }


def welch_interval(late, early, name):
    result = stats.ttest_ind(late.dropna(), early.dropna(), equal_var=False)
    low, high = result.confidence_interval(0.95)
    return {
        name: (low + high) / 2,
        f"{name}_se": (high - low) / 2 / stats.t.ppf(0.975, result.df),
        f"{name}_low": low,
        f"{name}_high": high,
    }


def paired_summary(df, label):
    row = dict(
        kind=label[0],
        years=label[1],
        split=label[2],
        panels=len(df),
        reviews=df.reviews.sum(),
        matched_reviews=df.matched_reviews.sum(),
    )
    for name in ("claims", "overlap"):
        ref, ours = df[f"{name}_ref"], df[f"{name}_model"]
        row.update({f"{name}_ref": ref.mean(), f"{name}_model": ours.mean()})
        row.update(mean_interval(ours - ref, f"{name}_bias"))
        row[f"{name}_rho"] = stats.spearmanr(ref, ours).statistic if ref.nunique() > 1 else np.nan
    for name in ("recall", "precision"):
        row.update(mean_interval(df[name], name))
    return row


def bias_change(late, early, label):
    row = dict(kind=label[0], years=label[1], split="all", panels=len(late) + len(early))
    for name in ("claims", "overlap"):
        bias = [df[f"{name}_model"] - df[f"{name}_ref"] for df in (late, early)]
        row.update(welch_interval(*bias, f"{name}_bias"))
    for name in ("recall", "precision"):
        row.update(welch_interval(late[name], early[name], name))
    return row


def reference(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "reference.tex",
    matcher=settings.GOLD_JUDGE_MODEL,
):
    data, gold = Path(data), Path(gold)
    matches = load_jsonl(data / f"match_{model}" / f"clusters_{matcher}.jsonl", "panel_id")
    ref_claims = {rid: row["claims"] for rid, row in load_jsonl(gold / "adjudicate" / "out.jsonl", "review_id").items()}
    ref_clusters = load_jsonl(gold / "adjudicate2" / "out.jsonl", "panel_id")
    model_claims = {rid: row["claims"] for rid, row in load_jsonl(data / f"claims_{model}.jsonl", "review_id").items()}
    model_clusters = load_jsonl(data / f"clusters_{model}.jsonl", "panel_id")
    rows = []
    for panel in read_jsonl(gold / "panels_tune.jsonl"):
        ids, ours = panel["review_ids"], model_clusters[panel["panel_id"]]
        assert sorted(ours["review_ids"]) == sorted(ids), panel["panel_id"]
        matched_ids = [r for r in ids if r in matches]
        matched = [matched_claims(matches[r]["clusters"], len(ref_claims[r]), len(model_claims[r])) for r in matched_ids]
        n_ref = sum(len(ref_claims[r]) for r in matched_ids)
        n_ours = sum(len(model_claims[r]) for r in matched_ids)
        rows.append(
            dict(
                kind=panel["kind"],
                year=panel["year"],
                split=panel["split"],
                claims_ref=np.mean([len(ref_claims[r]) for r in ids]),
                claims_model=np.mean([len(model_claims[r]) for r in ids]),
                overlap_ref=100 * strict_overlap(ids, ref_claims, ref_clusters[panel["panel_id"]]["clusters"]),
                overlap_model=100 * strict_overlap(ours["review_ids"], model_claims, ours["clusters"]),
                recall=100 * sum(m[0] for m in matched) / n_ref if n_ref else np.nan,
                precision=100 * sum(m[1] for m in matched) / n_ours if n_ours else np.nan,
                reviews=len(ids),
                matched_reviews=len(matched_ids),
            )
        )
    panels = pd.DataFrame(rows)
    summary = []
    for kind, by_kind in panels.groupby("kind", sort=False):
        summary += [paired_summary(cell, (kind, str(year), "all")) for year, cell in by_kind.groupby("year")]
        summary.append(paired_summary(by_kind, (kind, "all", "all")))
        summary.append(paired_summary(by_kind[by_kind.split == "test"], (kind, "all", "test")))
        last = by_kind.year.max()
        summary.append(bias_change(by_kind[by_kind.year == last], by_kind[by_kind.year < last], (kind, f"{last} minus earlier")))
    out = pd.DataFrame(summary)
    out.round(3).to_csv(Path(reports) / f"reference_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Claims per review} & \multicolumn{3}{c}{Overlap (\%)} & \multicolumn{2}{c}{Matched (\%)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}",
        r"Panels & Years & $n$ & Ref. & Luna & Diff. (SE) & Ref. & Luna & Diff. (SE) & Recall & Precision \\",
        r"\midrule",
    ]
    names = {"same": "Same-paper", "cross": "Cross-paper"}
    for r in out[out.split == "all"].itertuples():
        years = r.years.replace(" minus earlier", r" $-$ earlier").replace("all", "All")
        if "earlier" in years:
            cells = f"& & {r.claims_bias:+.2f} ({r.claims_bias_se:.2f}) & & & {r.overlap_bias:+.1f} ({r.overlap_bias_se:.1f})"
            matched = f"{r.recall:+.1f} ({r.recall_se:.1f}) & {r.precision:+.1f} ({r.precision_se:.1f})"
        else:
            cells = (
                f"{r.claims_ref:.1f} & {r.claims_model:.1f} & {r.claims_bias:+.2f} ({r.claims_bias_se:.2f}) & "
                f"{r.overlap_ref:.1f} & {r.overlap_model:.1f} & {r.overlap_bias:+.1f} ({r.overlap_bias_se:.1f})"
            )
            matched = f"{r.recall:.1f} ({r.recall_se:.1f}) & {r.precision:.1f} ({r.precision_se:.1f})"
        lines.append(f"{names[r.kind]} & {years} & {r.panels} & {cells} & {matched} \\\\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


if __name__ == "__main__":
    fire.Fire(
        {
            "robust": robust,
            "forms": forms,
            "detector_placebo": detector_placebo,
            "rq1_evidence": rq1_evidence,
            "journal_gaps": journal_gaps,
            "reference": reference,
        }
    )
