import re
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm

import ai
import extract
import hub
from between import normalize_rows, openai_embed
from jsonl import load_jsonl, read_jsonl
from metrics import assign, mean_overlap, review_groups
from metrics import overlap as pair_overlap
from outcomes import excluded_review_ids, in_window
from settings import settings

REMOVED_FIELD_TAGS = ("_nocq", "_noq")
HALF_SPAN = 0.5
LABEL_FIELDS = ("category", "target", "specific")
PAIR_COUNTS = ["pair_tp", "pair_fp", "pair_fn", "pair_candidates"]
COVER_QUANTILE = 0.95
MIN_DIRECT_PAPERS = 10
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
    "paper_summary": "summary",
    "critical_strengths": "critical strengths",
    "critical_weaknesses": "critical weaknesses",
    "questions_for_authors": "questions",
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
    variants = [
        ("Main overlap rate", main, "hit_rate_strict"),
        ("Specific claims only", specific, "hit_spec"),
        ("Same part of paper (lenient)", main, "hit_rate_lenient"),
        ("3-review subsets, grouped again", sub3, "hit_rate_strict"),
        ("Questions field removed", noq, "hit_rate_strict"),
    ]
    checks = [(*v, 2025, 2026) for v in variants]
    checks.append(("Withdrawn 2026 papers added", with_wd, "hit_rate_strict", 2025, 2026))
    checks += [(*v, 2024 if v[1] is noq else 2023, settings.ICLR_LAST) for v in variants]
    rows = []
    for name, df, col, base_year, last_year in checks:
        base = df[df.year == base_year][col].dropna() * 100
        last = df[df.year == last_year][col].dropna() * 100
        rows.append(
            {
                "check": name,
                "base_year": base_year,
                "base": base.mean(),
                "last_year": last_year,
                "last": last.mean(),
                "change": last.mean() - base.mean(),
                "se": np.hypot(base.sem(), last.sem()),
            }
        )
    out = pd.DataFrame(rows)
    out.round(3).to_csv(Path(reports) / f"robust_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Check & Base year & Base & Last year & Last & Change (SE) \\",
        r"\midrule",
    ]
    for k, r in enumerate(out.itertuples()):
        if k and r.last_year == settings.ICLR_LAST and out.last_year.iloc[k - 1] != settings.ICLR_LAST:
            lines.append(r"\midrule")
        lines.append(
            f"{r.check} & {r.base_year} & {r.base:.1f} & {r.last_year} & {r.last:.1f} & {r.change:+.1f} ({r.se:.1f}) \\\\"
        )
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
        f"conf == 'ICLR' and year >= {settings.ICLR_FROM} and n_claims > 0"
    )
    clusters = hub.clusters(model, panel_ids=main.panel_id)
    claims = hub.panel_claims(model, clusters)
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
    reviews = hub.corpus_reviews(["conf", "year", "form"]).query("conf == 'ICLR'")
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


def meld_flag_at(meld, quantile):
    threshold = meld[meld.year <= settings.MELD_CALIBRATION_LAST_YEAR].score.quantile(quantile)
    return (meld.score > threshold).astype(int)


def detector_placebo(model=settings.DEFAULT_MODEL, share=0.45, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    iclr_pairs = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    iclr_pairs = iclr_pairs[iclr_pairs.conf == "ICLR"]
    scores = hub.meld_flags()
    periods = {
        str(settings.ICLR_FROM): (settings.ICLR_FROM, settings.ICLR_FROM),
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


def review_chars(ids):
    return pd.Series({review_id: len(text) for review_id, text in hub.texts(ids).items()})


def pair_design(model):
    design = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    chars = review_chars(set(design.a) | set(design.b))
    design["log_short"], design["log_long"] = short_long_logs(chars, design)
    return design


def rq1_evidence(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    data, reports = Path(data), Path(reports)
    meld = hub.meld_flags()
    flag, written = meld.flag.astype(int), meld.written.astype(int)
    design = pair_design(model)
    claim_counts = pd.read_parquet(data / f"claims_table_{model}.parquet", columns=["review_id"]).groupby("review_id").size()
    design["claims_short"], design["claims_long"] = short_long_logs(claim_counts, design)
    iclr = design[(design.conf == "ICLR") & (design.year >= 2024)]
    rows = []

    def add(group, label, est, err, papers):
        rows.append(dict(group=group, label=label, est=est, se=err, papers=papers))

    pipe = "Pipeline"
    for label, subset_pairs, pair_flags, controls in [
        ("ICLR 2024–27", iclr, flag, ()),
        ("+ length and claim count", iclr.dropna(), flag, ("log_short", "log_long", "claims_short", "claims_long")),
        ("NeurIPS 2024–25", design[(design.conf == "NeurIPS") & (design.year >= 2024)], flag, ()),
        ("ICML 2025–26", design[design.conf == "ICML"], flag, ()),
        ("LLM-written, ICLR", iclr, written, ()),
        ("99th-percentile threshold, ICLR", iclr, meld_flag_at(meld, 0.99), ()),
        ("99.5th-percentile threshold, ICLR", iclr, meld_flag_at(meld, 0.995), ()),
    ]:
        gap = fe_gap(subset_pairs, pair_flags, controls)
        add(pipe, label, gap["two"], gap["two_se"], subset_pairs.panel_id.nunique())
    for year, year_pairs in iclr.groupby("year"):
        gap = fe_gap(year_pairs, flag)
        add("ICLR by year", f"ICLR {year}", gap["two"], gap["two_se"], year_pairs.panel_id.nunique())
    placebo = pd.read_csv(reports / f"detector_placebo_{model}.csv")
    for r in placebo.itertuples():
        add("Placebo before ChatGPT", f"Top-45% scores, ICLR {r.period}", r.two, r.two_se, r.papers)
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"rq1_evidence_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def journal_gaps(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR, first_year=2024):
    flag = hub.meld_flags().flag.astype(int)
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    journals = pairs[~pairs.conf.isin(settings.ML_VENUES) & (pairs.year >= first_year)]
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


def pair_counts(pairs, flag):
    return pairs.a.map(flag).fillna(0).astype(int) + pairs.b.map(flag).fillna(0).astype(int)


def identification_row(label, pairs, flag):
    pairs = pairs.assign(n_flagged=pair_counts(pairs, flag).to_numpy())
    reports = pd.concat(
        [pairs[["panel_id", "a"]].rename(columns={"a": "r"}), pairs[["panel_id", "b"]].rename(columns={"b": "r"})]
    )
    n_reports = reports.drop_duplicates().groupby("panel_id").size()
    types = pairs.groupby("panel_id").n_flagged.nunique()
    by_type = pairs.groupby(["panel_id", "n_flagged"]).hit.mean().unstack()
    endpoint = (by_type.get(2) - by_type.get(0)).dropna() * 100 if {0, 2} <= set(by_type.columns) else pd.Series(dtype=float)
    gap = fe_gap(pairs, flag)
    row = dict(
        label=label,
        papers=len(n_reports),
        pairs=len(pairs),
        **{f"reports_{k}": int((n_reports.clip(upper=5) == k).sum()) for k in (2, 3, 4, 5)},
        varying=int((types >= 2).sum()),
        with_both_pair=int(pairs[pairs.n_flagged == 2].panel_id.nunique()),
        endpoint_papers=len(endpoint),
        **gap,
    )
    row.update(mean_interval(endpoint, "endpoint") if len(endpoint) > 1 else {})
    return row


PAIR_TYPES = ("HL", "LL", "WW")


def pair_type(pairs, flag, written):
    return pair_counts(pairs, flag).map({0: "HH", 1: "HL", 2: "LL"}).where(pair_counts(pairs, written) < 2, "WW")


def venue_pairs(pairs):
    window = pairs[pairs.conf.isin(settings.ML_VENUES) & (pairs.year >= 2024)]
    return list(window.groupby("conf"))


def pair_types(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    meld = hub.meld_flags()
    flag, written = meld.flag.astype(int), meld.written.astype(int)
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    rows = []
    for conf, cell in venue_pairs(pairs):
        types = pair_type(cell, flag, written).to_numpy()
        X = pd.DataFrame({t: (types == t) * 1.0 for t in PAIR_TYPES} | {"hit": cell.hit.to_numpy() * 100}, index=cell.index)
        X = X - X.groupby(cell.panel_id.to_numpy()).transform("mean")
        fit = sm.OLS(X.hit, X[list(PAIR_TYPES)]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(cell.panel_id)[0]})
        for t in PAIR_TYPES:
            rows.append(
                dict(conf=conf, pair_type=t, est=fit.params[t], se=fit.bse[t], papers=cell[types == t].panel_id.nunique())
            )
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / f"pair_types_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


PAIR_SIZE_OUTCOMES = {"hit": "Overlap", "shared": "Shared", "smaller": "Smaller set", "union": "Union"}
PAIR_TYPE_NAMES = {
    "HH": "Neither flagged",
    "HL": "One flagged",
    "LL": "Both flagged, not both LLM-written",
    "WW": "Both LLM-written",
}


def type_fit(cell, types, outcome, controls=()):
    X = pd.DataFrame({t: (types == t) * 1.0 for t in PAIR_TYPES} | {"y": cell[outcome].to_numpy()}, index=cell.index)
    X = X.join(cell[list(controls)])
    X = X - X.groupby(cell.panel_id.to_numpy()).transform("mean")
    return sm.OLS(X.y, X[[*PAIR_TYPES, *controls]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(cell.panel_id)[0]})


def pair_type_sizes(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "pair_type_sizes.tex",
):
    meld = hub.meld_flags()
    flag, written = meld.flag.astype(int), meld.written.astype(int)
    pairs = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet"))
    cell = pairs[(pairs.conf == "ICLR") & (pairs.year >= 2024)].assign(hit=lambda d: d.hit * 100)
    types = pair_type(cell, flag, written).to_numpy()
    fits = {name: type_fit(cell, types, name) for name in PAIR_SIZE_OUTCOMES}
    fits["shared_adjusted"] = type_fit(cell, types, "shared", ("smaller", "larger"))
    rows = []
    for t in ("HH", *PAIR_TYPES):
        row = dict(pair_type=t, pairs=int((types == t).sum()))
        for name, fit in fits.items():
            if t == "HH":
                row[name] = cell[types == "HH"][name.replace("_adjusted", "")].mean()
            else:
                row[name], row[name + "_se"] = fit.params[t], fit.bse[t]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / f"pair_type_sizes_{model}.csv", index=False)
    print(out.round(3).to_string(index=False))
    lines = [
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.7cm}rrrrrr}",
        r"\toprule",
        r"Pair type & Pairs & \shortstack{Overlap\\(points)} & Shared & \shortstack{Smaller\\set} & Union & \shortstack{Shared,\\size-adj.} \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        if r.pair_type == "HH":
            cells = [f"{r.hit:.1f}", f"{r.shared:.2f}", f"{r.smaller:.2f}", f"{r.union:.2f}", f"{r.shared:.2f}"]
        else:
            cells = [
                f"{getattr(r, name):+.{1 if name == 'hit' else 2}f} ({getattr(r, name + '_se'):.{1 if name == 'hit' else 2}f})"
                for name in [*PAIR_SIZE_OUTCOMES, "shared_adjusted"]
            ]
        lines.append(f"{PAIR_TYPE_NAMES[r.pair_type]} & {r.pairs:,} & " + " & ".join(cells) + r" \\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


DETECTOR_QUANTILES = (0.99, 0.995, 0.999)


def detector_row(label, quantile, flags, pairs, iclr_last):
    pre = flags[flags.year <= settings.MELD_CALIBRATION_LAST_YEAR]
    flag = flags[~flags.index.duplicated()].flag.astype(int)
    counts = pair_counts(pairs, flag)
    gap = fe_gap(pairs, flag)
    varying = (counts.groupby(pairs.panel_id.to_numpy()).nunique() > 1).sum()
    last = flag.reindex(iclr_last)
    return dict(
        label=label,
        percentile=quantile * 100,
        pre_max=pre.groupby("conf").flag.mean().max() * 100,
        iclr_last=last.mean() * 100,
        two=gap["two"],
        two_se=gap["two_se"],
        papers=int(varying),
        two_pairs=int((counts == 2).sum()),
    )


def detector_choices(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "detector_choices.tex",
):
    meld = hub.meld_flags(["conf", "year", "score", "task"])
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    pairs = pairs[(pairs.conf == "ICLR") & (pairs.year >= 2024)]
    iclr_last = meld[(meld.conf == "ICLR") & (meld.year == pairs.year.max())].index
    iclr_last = iclr_last[~iclr_last.isin(excluded_review_ids())]
    calibration = meld[meld.year <= settings.MELD_CALIBRATION_LAST_YEAR].score
    rows = []
    for quantile in DETECTOR_QUANTILES:
        flagged = meld.assign(flag=meld.score > calibration.quantile(quantile))
        written = flagged.assign(flag=flagged.flag & (flagged.task == settings.MELD_WRITTEN_TASK))
        rows.append(detector_row("Flagged", quantile, flagged, pairs, iclr_last))
        rows.append(detector_row("LLM-written", quantile, written, pairs, iclr_last))
    out = pd.DataFrame(rows).sort_values(["label", "percentile"], key=lambda s: s.eq("LLM-written") if s.name == "label" else s)
    out.to_csv(Path(reports) / f"detector_choices_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Label & Percentile & Pre-2023 max (\%) & ICLR "
        + str(pairs.year.max())
        + r" (\%) & Two vs.\ none (SE) & Two pairs & Papers \\",
        r"\midrule",
    ]
    for k, r in enumerate(out.itertuples()):
        if k and r.percentile == DETECTOR_QUANTILES[0] * 100:
            lines.append(r"\midrule")
        lines.append(
            f"{r.label} & {r.percentile:g} & {r.pre_max:.2f} & {r.iclr_last:.1f} & {r.two:+.1f} ({r.two_se:.1f}) & {r.two_pairs:,} & {r.papers:,} \\\\"
        )
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


JOURNAL_SUBSETS = ["Identifying", "3--5 reports, other", "Two reports"]
JOURNAL_ORDER = ["F1000", "PLOS", "Copernicus", "BMC-Med", "SciPost"]


def journal_subsets(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "journal_subsets.tex",
):
    data = Path(data)
    flag = hub.meld_flags().flag.astype(int)
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    journals = pairs[~pairs.conf.isin(settings.ML_VENUES) & (pairs.year >= 2024)]
    journals = journals.assign(n_flagged=pair_counts(journals, flag).to_numpy())
    texts = pd.concat(
        [journals[["panel_id", "a"]].rename(columns={"a": "r"}), journals[["panel_id", "b"]].rename(columns={"b": "r"})]
    ).drop_duplicates()
    chars = review_chars(set(texts.r))
    wanted = set(texts.r)
    claims = pd.Series({r: len(c) for r, c in hub.claims(model, wanted).items()})
    texts = texts.assign(
        chars=texts.r.map(chars) / 1000, claims=texts.r.map(claims).fillna(0), flagged=texts.r.map(flag).fillna(0) * 100
    )
    papers = journals.groupby("panel_id").agg(
        conf=("conf", "first"), year=("year", "first"), overlap=("hit", "mean"), types=("n_flagged", "nunique")
    )
    papers = papers.join(
        texts.groupby("panel_id").agg(
            n=("r", "size"), chars=("chars", "mean"), claims=("claims", "mean"), flagged=("flagged", "mean")
        )
    )
    papers["overlap"] *= 100
    papers["subset"] = np.select([papers.types >= 2, papers.n >= 3], JOURNAL_SUBSETS[:2], JOURNAL_SUBSETS[2])
    rows = []
    for subset, cell in [*((s, papers[papers.subset == s]) for s in JOURNAL_SUBSETS), ("All", papers)]:
        shares = cell.conf.value_counts(normalize=True).reindex(JOURNAL_ORDER, fill_value=0) * 100
        rows.append(
            dict(
                subset=subset,
                papers=len(cell),
                year=cell.year.mean(),
                **{c: cell[c].mean() for c in ("n", "chars", "claims", "overlap", "flagged")},
                overlap_se=cell.overlap.sem(),
                **{f"share_{c}": shares[c] for c in JOURNAL_ORDER},
            )
        )
    out = pd.DataFrame(rows)
    out.round(3).to_csv(Path(reports) / f"journal_subsets_{model}.csv", index=False)
    print(out.round(2).T.to_string())
    names = {"Identifying": "Ident.", "3--5 reports, other": "3--5", "Two reports": "Two"}
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        " & ".join(["", *(names.get(x, x) for x in out.subset)]) + r" \\",
        r"\midrule",
    ]
    cells = [
        ("Papers", lambda r: f"{int(r.papers):,}"),
        ("Year", lambda r: f"{r.year:.1f}"),
        ("Reports", lambda r: f"{r.n:.1f}"),
        ("kchars", lambda r: f"{r.chars:.1f}"),
        ("Claims", lambda r: f"{r.claims:.1f}"),
        (r"Overlap (\%)", lambda r: f"{r.overlap:.1f} ({r.overlap_se:.1f})"),
        (r"Flagged (\%)", lambda r: f"{r.flagged:.1f}"),
        *((rf"{c} (\%)", lambda r, c=c: f"{r[f'share_{c}']:.0f}") for c in JOURNAL_ORDER),
    ]
    for label, cell in cells:
        lines.append(" & ".join([label.replace("BMC-Med", "BMC"), *(cell(r) for _, r in out.iterrows())]) + r" \\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def identification(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "identification.tex",
):
    meld = hub.meld_flags()
    flag, written = meld.flag.astype(int), meld.written.astype(int)
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    iclr = pairs[(pairs.conf == "ICLR") & (pairs.year >= 2024)]
    journals = pairs[~pairs.conf.isin(settings.ML_VENUES) & (pairs.year >= 2024)]
    cells = [
        ("ICLR 2024–27", iclr, flag),
        ("ICLR 2024–27, LLM-written", iclr, written),
        ("NeurIPS 2024–25", pairs[(pairs.conf == "NeurIPS") & (pairs.year >= 2024)], flag),
        ("ICML 2025–26", pairs[pairs.conf == "ICML"], flag),
        *[(f"{source} 2024–26", cell, flag) for source, cell in journals.groupby("conf")],
        ("All journals 2024–26", journals, flag),
    ]
    out = pd.DataFrame([identification_row(label, cell, f) for label, cell, f in cells])
    out.round(3).to_csv(Path(reports) / f"identification_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Papers by reports} & \multicolumn{2}{c}{Papers identifying} & \multicolumn{2}{c}{Two vs.\ none (pp)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Sample & Papers & 2 & 3 & 4+ & FE & Direct & FE (SE) & Direct [95\% CI] \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        direct = (
            f"{r.endpoint:+.1f} [{r.endpoint_low:+.1f}, {r.endpoint_high:+.1f}]"
            if r.endpoint_papers >= MIN_DIRECT_PAPERS
            else "--"
        )
        lines.append(
            f"{r.label} & {r.papers:,} & {r.reports_2:,} & {r.reports_3:,} & {r.reports_4 + r.reports_5:,} & "
            f"{r.varying:,} & {r.endpoint_papers:,} & {r.two:+.1f} ({r.two_se:.1f}) & {direct} \\\\"
        )
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def coverage_panel(panel, claims):
    df = assign(panel, claims)
    n = len(panel["review_ids"])
    groups = review_groups([], [], n) if df.empty else review_groups(df.review, df.cluster, n)
    union = set().union(*groups)
    new = [len(g - set().union(*(h for j, h in enumerate(groups) if j != i))) for i, g in enumerate(groups)]
    shared = [len(a & b) for a, b in combinations(groups, 2)]
    return dict(
        problems=len(union),
        per_review=np.mean([len(g) for g in groups]),
        new_by_third=np.mean(new),
        shared_per_pair=np.mean(shared),
        overlap=mean_overlap(groups),
    )


def coverage(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "coverage.tex",
):
    data = Path(data)
    sub3 = pd.read_parquet(settings.ANALYSIS_DIR / f"sub3_outcomes_{model}.parquet").query(
        f"conf == 'ICLR' and year >= {settings.ICLR_FROM}"
    )
    clusters = hub.clusters(model, panel_ids=sub3.panel_id)
    claims = hub.panel_claims(model, clusters)
    panels = pd.DataFrame(
        [dict(year=p.year, **coverage_panel(clusters[p.panel_id], claims)) for p in tqdm(sub3.itertuples(), total=len(sub3))]
    )
    cols = ["problems", "per_review", "new_by_third", "shared_per_pair", "overlap"]
    by_year = panels.groupby("year")[cols]
    out = pd.concat([by_year.mean(), by_year.sem().add_suffix("_se")], axis=1)
    out[["overlap", "overlap_se"]] *= 100
    out.insert(0, "papers", panels.groupby("year").size())
    out = out.reset_index()
    out.round(3).to_csv(Path(reports) / f"coverage_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Year & Papers & Distinct problems & Problems per review & New from a third review & Shared per pair & Overlap (\%) \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        cells = " & ".join(
            f"{getattr(r, c):.{1 if c == 'overlap' else 2}f} ({getattr(r, c + '_se'):.{1 if c == 'overlap' else 2}f})"
            for c in cols
        )
        lines.append(f"{r.year} & {r.papers:,} & {cells} \\\\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


PASS1_SAMPLES = {"ml": ["main", "icml", "withdrawn"], "journal": ["journal"]}
PASS1_VARIANTS = {"ml": ["noq", "nocq"], "journal": []}


def repro(model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR):
    values = {}
    for kind, names in PASS1_SAMPLES.items():
        values[f"pass1_{kind}_reviews"] = sum(len(hub.sample(name, ["review_id"])) for name in names) + sum(
            len(hub.variant(name, ["review_id"])) for name in PASS1_VARIANTS[kind]
        )
    values.update(hub.extraction_counts(model))
    pairs = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["a", "b", "conf", "year"]))
    flags = hub.meld_flags(["year"])
    calibration = flags[flags.year <= settings.MELD_CALIBRATION_LAST_YEAR]
    in_analysis = calibration.index.isin(set(pairs.a) | set(pairs.b))
    values["calibration_reviews"] = len(calibration)
    values["calibration_in_analysis"] = int(in_analysis.sum())
    values["calibration_outside"] = int((~in_analysis).sum())
    out = pd.Series(values)
    out.to_csv(Path(reports) / f"repro_{model}.csv", header=["value"], index_label="name")
    print(out.round(2).to_string())


def position_overlaps(panel, claims):
    df = assign(panel, claims)
    n = len(panel["review_ids"])
    groups = review_groups([], [], n) if df.empty else review_groups(df.review, df.cluster, n)
    return {(i, j): pair_overlap(groups[i], groups[j]) for i, j in combinations(range(n), 2)}


def rewrite_pairs(out, claims, clusters, main, panel_prefix):
    rows = []
    for p in read_jsonl(out / "papers.jsonl"):
        pid = p["panel_id"]
        if f"rw0_{pid}" not in clusters or f"{panel_prefix}{pid}" not in clusters:
            continue
        rewritten = [r in p["rewritten"] for r in p["review_ids"]]
        before = position_overlaps(clusters[f"rw0_{pid}"], claims)
        after = position_overlaps(clusters[f"{panel_prefix}{pid}"], claims)
        first_run = position_overlaps(main[pid], claims)
        group = "ICLR" if p["conf"] == "ICLR" else "Journals"
        for (i, j), hit in before.items():
            common = dict(paper=pid, group=group, pair=f"{pid}:{i}-{j}")
            rows.append(dict(common, version=0, n_rw=0, hit=hit, hit_first=first_run[(i, j)]))
            rows.append(dict(common, version=2, n_rw=rewritten[i] + rewritten[j], hit=after[(i, j)], hit_first=np.nan))
    return pd.DataFrame(rows)


def pair_fe(df):
    X = pd.DataFrame(
        {
            "one": (df.n_rw == 1) * 1.0,
            "two": (df.n_rw == 2) * 1.0,
            "regrouped": (df.version == 2) * 1.0,
            "hit": df.hit * 100,
        },
        index=df.index,
    )
    X = X - X.groupby(df.pair.to_numpy()).transform("mean")
    fit = sm.OLS(X.hit, X[["one", "two", "regrouped"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.paper)[0]})
    return {name: fit.params[name] for name in ("one", "two")} | {f"{name}_se": fit.bse[name] for name in ("one", "two")}


def preservation(clusters, claims, preserve_prefix):
    rows = []
    for panel_id, panel in clusters.items():
        if not panel_id.startswith(preserve_prefix):
            continue
        original, rewritten = panel["review_ids"]
        n_orig, n_rw = len(claims.get(original, [])), len(claims.get(rewritten, []))
        hit_orig, hit_rw = matched_claims(panel["clusters"], n_orig, n_rw)
        rows.append(hit_orig / n_orig * 100 if n_orig else np.nan)
    return pd.Series(rows, dtype=float)


def standalone_change(clusters, claims, papers, panel_prefix):
    rows = []
    for p in papers:
        before, after = (
            clusters.get(f"{settings.REWRITE_PAIR_PREFIX}rw0_{p['panel_id']}"),
            clusters.get(f"{settings.REWRITE_PAIR_PREFIX}{panel_prefix}{p['panel_id']}"),
        )
        if before and after:
            group = "ICLR" if p["conf"] == "ICLR" else "Journals"
            change = strict_overlap(after["review_ids"], claims, after["clusters"]) - strict_overlap(
                before["review_ids"], claims, before["clusters"]
            )
            rows.append(dict(group=group, change=100 * change))
    return pd.DataFrame(rows)


REWRITE_LABELS = {"outline": "Outline", "polish": "Polished", "outline_second": "Outline", "polish_second": "Polished"}
REWRITE_TABLE_ORDER = ["outline", "outline_second", "polish", "polish_second"]
WRITER_LABELS = {settings.AI_REFERENCE_MODEL: "Luna", settings.REWRITE_SECOND_WRITER: "Kimi"}


def rewrite_test(
    model=settings.DEFAULT_MODEL,
    matcher=settings.REWRITE_AUDIT_MODEL,
    out=settings.REWRITE_TEST_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "rewrite_test.tex",
):
    out = Path(out)
    claims = {rid: row["claims"] for rid, row in load_jsonl(out / f"claims_{model}.jsonl", "review_id").items()}
    clusters = load_jsonl(out / f"clusters_{model}.jsonl", "panel_id")
    papers = list(read_jsonl(out / "papers.jsonl"))
    main = hub.clusters(model, panel_ids={p["panel_id"] for p in papers})
    audited = {p["panel_id"] for p in read_jsonl(out / "panels_audit.jsonl")}
    audit = {pid: row for pid, row in load_jsonl(out / f"clusters_{matcher}.jsonl", "panel_id").items() if pid in audited}
    rows, all_pairs = [], []
    for condition, c in settings.REWRITE_CONDITIONS.items():
        pairs = rewrite_pairs(out, claims, clusters, main, c.panel).assign(condition=condition)
        all_pairs.append(pairs)
        luna_kept = preservation(clusters, claims, c.preserve)
        audit_kept = preservation(audit, claims, c.preserve)
        rewritten = [r for p in papers for r in p["rewritten"] if c.text + r in claims]
        alone = standalone_change(clusters, claims, papers, c.panel)
        for group, cell in [*pairs.groupby("group"), ("All", pairs)]:
            alone_cell = alone if group == "All" else alone[alone.group == group]
            two_pairs = cell[cell.pair.isin(cell[cell.n_rw == 2].pair)]
            wide = two_pairs.pivot(index="pair", columns="version", values="hit") * 100
            kept_pairs = cell[cell.pair.isin(cell[(cell.version == 2) & (cell.n_rw == 0)].pair)]
            unchanged = kept_pairs.pivot(index="pair", columns="version", values="hit") * 100
            first = cell[cell.version == 0]
            rows.append(
                dict(
                    condition=condition,
                    writer=c.writer,
                    group=group,
                    papers=cell.paper.nunique(),
                    pairs=cell.pair.nunique(),
                    two_before=wide[0].mean(),
                    two_after=wide[2].mean(),
                    **mean_interval(wide[2] - wide[0], "two_diff"),
                    **mean_interval(unchanged[2] - unchanged[0], "unchanged_diff"),
                    **mean_interval((first.hit - first.hit_first) * 100, "regroup_diff"),
                    **mean_interval(alone_cell.change, "alone_diff"),
                    **pair_fe(cell),
                    luna_recall=luna_kept.mean(),
                    audit_recall=audit_kept.mean(),
                    audit_reviews=int(audit_kept.notna().sum()),
                    claims_original=np.mean([len(claims.get(r, [])) for r in rewritten]),
                    claims_rewrite=np.mean([len(claims.get(c.text + r, [])) for r in rewritten]),
                )
            )
    pd.concat(all_pairs).to_parquet(settings.ANALYSIS_DIR / f"rewrite_pairs_{model}.parquet")
    table = pd.DataFrame(rows)
    table.round(3).to_csv(Path(reports) / f"rewrite_test_{model}.csv", index=False)
    print(table.round(2).T.to_string())
    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{Both changed} & & \\",
        r"\cmidrule(lr){5-7}",
        r"Condition & Writer & Source & Pairs & Before & After & \shortstack{Change\\{[95\% CI]}} & \shortstack{FE\\(SE)} & \shortstack{Alone\\{[95\% CI]}} \\",
        r"\midrule",
    ]
    groups = table.groupby("condition").group.nunique()
    ordered = table.set_index("condition").loc[REWRITE_TABLE_ORDER].reset_index()
    for k, r in enumerate(ordered.itertuples()):
        if groups[r.condition] == 2 and r.group == "All":
            continue
        if k and r.group == "ICLR":
            lines.append(r"\midrule")
        name, writer = (REWRITE_LABELS[r.condition], WRITER_LABELS[r.writer]) if r.group == "ICLR" else ("", "")
        lines.append(
            f"{name} & {writer} & {r.group} & {r.pairs:,} & {r.two_before:.1f} & {r.two_after:.1f} & "
            f"{r.two_diff:+.1f} [{r.two_diff_low:+.1f}, {r.two_diff_high:+.1f}] & {r.two:+.1f} ({r.two_se:.1f}) & "
            f"{r.alone_diff:+.1f} [{r.alone_diff_low:+.1f}, {r.alone_diff_high:+.1f}] \\\\"
        )
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def claim_group_ids(clusters):
    group_of = {}
    for g, cluster in enumerate(clusters):
        for cid in cluster["claim_ids"]:
            group_of.setdefault(cid, g)
    return group_of


def grouped(review_ids, claims, clusters):
    group_of = claim_group_ids(clusters)
    return [{group_of.get(f"R{j}-{k}", f"R{j}-{k}") for k in range(len(claims[rid]))} for j, rid in enumerate(review_ids)]


def strict_overlap(review_ids, claims, clusters):
    return mean_overlap(grouped(review_ids, claims, clusters))


def matched_claims(clusters, n_ref, n_ours):
    valid = {f"R0-{k}" for k in range(n_ref)} | {f"R1-{k}" for k in range(n_ours)}
    hit = set()
    for cluster in clusters:
        ids = valid.intersection(cluster["claim_ids"])
        if {cid[:2] for cid in ids} == {"R0", "R1"}:
            hit |= ids
    return sum(cid.startswith("R0") for cid in hit), sum(cid.startswith("R1") for cid in hit)


def pair_detection_summary(panels):
    counts = panels[PAIR_COUNTS].to_numpy()
    row = dict(zip(PAIR_COUNTS, counts.sum(axis=0)))
    rng = np.random.default_rng(settings.REFERENCE_BOOTSTRAP_SEED)
    draws = rng.integers(len(counts), size=(settings.REFERENCE_BOOTSTRAPS, len(counts)))
    tp, fp, fn, candidates = np.vstack([counts.sum(axis=0), counts[draws].sum(axis=1)]).T
    for name, numerator, denominator in [
        ("precision", tp, tp + fp),
        ("recall", tp, tp + fn),
        ("f1", 2 * tp, 2 * tp + fp + fn),
        ("accuracy", candidates - fp - fn, candidates),
    ]:
        scores = np.divide(100.0 * numerator, denominator, out=np.full(len(tp), np.nan), where=denominator > 0)
        valid = scores[1:][np.isfinite(scores[1:])]
        low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
        row.update({f"pair_{name}": scores[0], f"pair_{name}_low": low, f"pair_{name}_high": high})
    return row


def mean_interval(values, name):
    values = values.dropna()
    half = stats.t.ppf(0.975, len(values) - 1) * values.sem()
    return {
        name: values.mean(),
        f"{name}_se": values.sem(),
        f"{name}_low": values.mean() - half,
        f"{name}_high": values.mean() + half,
    }


def report_cells(df):
    cells = [(dict(conf="all", years="all", stratum="all", split="all"), df)]
    cells += [
        (dict(conf="all", years="all", stratum="all", split=split), cell) for split, cell in df.groupby("split", sort=False)
    ]
    cells += [(dict(conf=conf, years="all", stratum="all", split="all"), cell) for conf, cell in df.groupby("conf", sort=False)]
    by_year = df[df.conf.isin(settings.ML_VENUES)].groupby(["conf", "year"], sort=False)
    cells += [(dict(conf=conf, years=str(year), stratum="all", split="all"), cell) for (conf, year), cell in by_year]
    cells += [
        (dict(conf="all", years="all", stratum=name, split="all"), cell) for name, cell in df.groupby("stratum", sort=False)
    ]
    return cells


def write_cell_table(out, tex, columns):
    names = {"same": "Same-paper", "cross": "Cross-paper"}
    header = "Panels & Source & Years or flags & $n$ & " + " & ".join(title for _, title, _, _ in columns) + r" \\"
    lines = [r"\begin{tabular}{lll" + "r" * (len(columns) + 1) + "}", r"\toprule", header, r"\midrule"]
    for r in out[out.split == "all"].itertuples():
        cell = (r.years if r.stratum == "all" else r.stratum).replace("all", "All")
        values = []
        for name, _, digits, with_se in columns:
            text = f"{getattr(r, name):.{digits}f}"
            values.append(text + f" ({getattr(r, name + '_se'):.{digits}f})" if with_se else text)
        lines.append(f"{names[r.kind]} & {r.conf.replace('all', 'All')} & {cell} & {r.panels} & " + " & ".join(values) + r" \\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def gold_dir(gold):
    path = Path(gold)
    return path if path.exists() else hub.root() / settings.HF_GOLD_DIR


def panel_meta(panel):
    return {key: panel[key] for key in ("panel_id", "kind", "conf", "year", "stratum", "split")}


def quote_spans(text, claims):
    norm = extract.normalize_quote(text)
    spans = []
    for claim in claims:
        quote = extract.normalize_quote(claim["quote"])
        start = norm.find(quote) if quote else -1
        first = norm.count(" ", 0, start)
        spans.append(frozenset(range(first, first + len(quote.split()))) if start >= 0 else frozenset())
    return spans


def span_f1(a, b):
    return 100 * 2 * len(a & b) / (len(a) + len(b)) if a or b else np.nan


def share_in(a, b):
    return 100 * len(a & b) / len(a) if a else np.nan


def token_f1(a, b):
    a, b = Counter(extract.normalize_quote(a).split()), Counter(extract.normalize_quote(b).split())
    return 100 * 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values()))


def aligned_claims(spans_a, claims_a, spans_b, claims_b):
    pairs = []
    for span, claim in zip(spans_a, claims_a):
        best, j = max(((len(span & other), j) for j, other in enumerate(spans_b)), default=(0, None))
        if span and best >= HALF_SPAN * len(span):
            pairs.append((claim, claims_b[j]))
    return pairs


def pair_scores(pairs, n_claims):
    row = {"aligned": 100 * len(pairs) / n_claims if n_claims else np.nan}
    for field in LABEL_FIELDS:
        row[field] = 100 * np.mean([a[field] == b[field] for a, b in pairs]) if pairs else np.nan
    row["claim_f1"] = np.mean([token_f1(a["claim"], b["claim"]) for a, b in pairs]) if pairs else np.nan
    return row


def covered(a, b, threshold):
    return 100 * ((a @ b.T).max(axis=1) >= threshold).mean() if len(a) and len(b) else np.nan


def within_list_similarities(vector_lists):
    sims = [(v @ v.T)[np.triu_indices(len(v), 1)] for v in vector_lists if len(v) > 1]
    return np.concatenate(sims)


def extraction_row(text, claims, vectors, threshold, model, council):
    spans = {m: quote_spans(text, claims[m]) for m in claims}
    words = {m: set().union(*spans[m]) for m in claims}
    counts = Counter(w for m in council for w in words[m])
    row = dict(claims_model=len(claims[model]), claims_council=np.mean([len(claims[m]) for m in council]))
    row["f1_model"] = pd.Series([span_f1(words[model], words[m]) for m in council]).mean()
    row["f1_council"] = pd.Series([span_f1(words[a], words[b]) for a, b in combinations(council, 2)]).mean()
    row["precision_pooled"] = share_in(words[model], set(counts))
    row["recall_majority"] = share_in({w for w, n in counts.items() if n >= 2}, words[model])
    row["recall_unanimous"] = share_in({w for w, n in counts.items() if n == len(council)}, words[model])
    model_pairs = [p for m in council for p in aligned_claims(spans[model], claims[model], spans[m], claims[m])]
    council_pairs = [p for a, b in permutations(council, 2) for p in aligned_claims(spans[a], claims[a], spans[b], claims[b])]
    row.update({f"{k}_model": v for k, v in pair_scores(model_pairs, len(council) * len(claims[model])).items()})
    n_council = sum(len(claims[a]) for a, _ in permutations(council, 2))
    row.update({f"{k}_council": v for k, v in pair_scores(council_pairs, n_council).items()})
    row["cover_recall_model"] = pd.Series([covered(vectors[m], vectors[model], threshold) for m in council]).mean()
    row["cover_precision_model"] = pd.Series([covered(vectors[model], vectors[m], threshold) for m in council]).mean()
    row["cover_council"] = pd.Series([covered(vectors[a], vectors[b], threshold) for a, b in permutations(council, 2)]).mean()
    return row


def claim_vectors(claims):
    texts = sorted({c["claim"] for by_review in claims.values() for cl in by_review.values() for c in cl})
    client = extract.openrouter_client()
    vectors = openai_embed(texts, settings.CLAIM_EMBED_MODEL, batch=64, threads=8, desc="embed claims", client=client)
    vectors = normalize_rows(vectors)
    index = {text: i for i, text in enumerate(texts)}
    return lambda cl: vectors[[index[c["claim"]] for c in cl]]


def extraction(
    model=settings.DEFAULT_MODEL,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "extraction.tex",
    council=settings.GOLD_COUNCIL,
):
    gold = gold_dir(gold)
    panels = list(read_jsonl(gold / "panels_tune.jsonl"))
    texts = pd.read_parquet(gold / "tune_reviews.parquet").set_index("review_id").text
    claims = {m: {} for m in council}
    for row in read_jsonl(gold / "claims_council.jsonl"):
        if "claims" in row:
            claims[row["model"]][row["review_id"]] = row["claims"]
    claims[model] = extract.load_claims(gold / f"claims_{model}.jsonl")
    vectors = claim_vectors(claims)
    scored = [(panel, rid) for panel in panels for rid in panel["review_ids"] if all(rid in claims[m] for m in claims)]
    review_vectors = {rid: {m: vectors(claims[m][rid]) for m in claims} for _, rid in scored}
    all_lists = [v for by_model in review_vectors.values() for v in by_model.values()]
    threshold = np.quantile(within_list_similarities(all_lists), COVER_QUANTILE)
    print(f"coverage threshold: cosine {threshold:.3f}")
    rows = []
    for panel, rid in scored:
        review_claims = {m: claims[m][rid] for m in claims}
        scores = extraction_row(texts[rid], review_claims, review_vectors[rid], threshold, model, council)
        rows.append(dict(**panel_meta(panel), review_id=rid, **scores))
    reviews = pd.DataFrame(rows)
    reviews.to_csv(Path(reports) / f"extraction_reviews_{model}.csv", index=False)
    keys = list(panel_meta(panels[0]))
    metrics = [c for c in reviews.columns if c not in keys + ["review_id"]]
    by_panel = reviews.groupby(keys, sort=False)
    panels = by_panel[metrics].mean().reset_index().assign(reviews=by_panel.size().to_numpy())
    summary = []
    for kind, by_kind in panels.groupby("kind", sort=False):
        for label, cell in report_cells(by_kind):
            row = dict(kind=kind, **label, panels=len(cell), reviews=int(cell.reviews.sum()), cover_threshold=threshold)
            for name in metrics:
                row.update(mean_interval(cell[name], name))
            summary.append(row)
    out = pd.DataFrame(summary)
    out.round(3).to_csv(Path(reports) / f"extraction_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    columns = [
        ("claims_model", "Claims Luna", 1, False),
        ("claims_council", "Claims council", 1, False),
        ("f1_model", "Span F1 Luna (SE)", 1, True),
        ("f1_council", "Span F1 council (SE)", 1, True),
        ("precision_pooled", r"Prec.\ pooled (SE)", 1, True),
        ("recall_unanimous", r"Rec.\ unanimous (SE)", 1, True),
        ("cover_recall_model", r"Cov.\ rec.\ Luna (SE)", 1, True),
        ("cover_precision_model", r"Cov.\ prec.\ Luna (SE)", 1, True),
        ("cover_council", r"Cov.\ council (SE)", 1, True),
    ]
    write_cell_table(out, tex, columns)


def panel_overlap(panel, claims, clusters):
    row = clusters.get(panel["panel_id"], {})
    if "clusters" not in row or any(rid not in claims for rid in panel["review_ids"]):
        return np.nan
    return 100 * strict_overlap(panel["review_ids"], claims, row["clusters"])


def pipelines(
    model=settings.DEFAULT_MODEL,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "pipelines.tex",
    contrasts_tex=settings.TABLES_DIR / "pipeline_contrasts.tex",
    council=settings.GOLD_COUNCIL,
):
    gold = gold_dir(gold)
    panels = list(read_jsonl(gold / "panels_tune.jsonl"))
    runs = {
        model: (extract.load_claims(gold / f"claims_{model}.jsonl"), load_jsonl(gold / f"clusters_{model}.jsonl", "panel_id"))
    }
    for member in council:
        runs[member] = (
            extract.load_claims(gold / f"claims_{member}.jsonl"),
            load_jsonl(gold / f"clusters_{member}_own.jsonl", "panel_id"),
        )
    rows = []
    for panel in panels:
        overlaps = {name: panel_overlap(panel, claims, clusters) for name, (claims, clusters) in runs.items()}
        if all(np.isfinite(v) for v in overlaps.values()):
            rows.append(dict(**panel_meta(panel), **{f"overlap_{name}": v for name, v in overlaps.items()}))
    panels = pd.DataFrame(rows)
    panels.to_csv(Path(reports) / f"pipelines_panels_{model}.csv", index=False)
    summary = []
    for kind, by_kind in panels.groupby("kind", sort=False):
        for label, cell in report_cells(by_kind):
            row = dict(kind=kind, **label, panels=len(cell))
            ours = cell[f"overlap_{model}"]
            members = [cell[f"overlap_{m}"] for m in council]
            row.update(mean_interval(ours, "overlap_model"))
            row.update(mean_interval(sum(members) / len(members), "overlap_council"))
            row.update(mean_interval(sum(ours - m for m in members) / len(members), "diff_model"))
            pair_diffs = [a - b for a, b in combinations(members, 2)]
            row.update(mean_interval(sum(d.abs() for d in pair_diffs) / len(pair_diffs), "absdiff_council"))
            row.update(mean_interval(sum((ours - m).abs() for m in members) / len(members), "absdiff_model"))
            row["corr_model"] = np.mean([ours.corr(m) for m in members]) if len(cell) > 2 else np.nan
            row["corr_council"] = np.mean([a.corr(b) for a, b in combinations(members, 2)]) if len(cell) > 2 else np.nan
            summary.append(row)
    out = pd.DataFrame(summary)
    out.round(3).to_csv(Path(reports) / f"pipelines_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    columns = [
        ("overlap_model", "Overlap Luna (SE)", 1, True),
        ("overlap_council", "Overlap council (SE)", 1, True),
        ("diff_model", r"Luna $-$ council (SE)", 1, True),
        ("absdiff_model", r"$|$Luna $-$ member$|$ (SE)", 1, True),
        ("absdiff_council", r"$|$member $-$ member$|$ (SE)", 1, True),
        ("corr_model", "Corr.\\ Luna", 2, False),
        ("corr_council", "Corr.\\ council", 2, False),
    ]
    write_cell_table(out, tex, columns)
    contrasts = pd.DataFrame([pipeline_contrasts(panels, name) for name in runs])
    contrasts.round(3).to_csv(Path(reports) / f"pipelines_contrasts_{model}.csv", index=False)
    print(contrasts.round(2).to_string(index=False))
    write_contrast_table(contrasts, contrasts_tex)


def bootstrap_interval(frame, statistic, name):
    rng = np.random.default_rng(settings.REFERENCE_BOOTSTRAP_SEED)
    draws = rng.integers(len(frame), size=(settings.REFERENCE_BOOTSTRAPS, len(frame)))
    values = np.array([statistic(frame.iloc[draw]) for draw in draws])
    values = values[np.isfinite(values)]
    low, high = np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)
    return {name: statistic(frame), f"{name}_low": low, f"{name}_high": high}


def pipeline_contrasts(panels, name):
    column = f"overlap_{name}"
    same = panels[panels.kind == "same"]
    iclr = same[same.conf == "ICLR"]
    first, last = settings.CONTRAST_YEARS

    def gap(frame, a, b):
        return frame[frame.stratum == a][column].mean() - frame[frame.stratum == b][column].mean()

    def slope(frame):
        return np.polyfit(frame.year.to_numpy(float), frame[column].to_numpy(float), 1)[0]

    def step(frame):
        return frame[frame.year == last][column].mean() - frame[frame.year == first][column].mean()

    row = dict(pipeline=name, panels=len(same), level=same[column].mean(), cross=panels[panels.kind == "cross"][column].mean())
    row.update(bootstrap_interval(same, lambda f: gap(f, "flagged", "plain"), "flag_gap"))
    row.update(bootstrap_interval(same, lambda f: gap(f, "written", "plain"), "written_gap"))
    row.update(bootstrap_interval(iclr, slope, "iclr_slope"))
    row.update(bootstrap_interval(iclr, step, "iclr_step"))
    return row


def write_contrast_table(contrasts, tex):
    first, last = settings.CONTRAST_YEARS
    columns = [
        ("flag_gap", r"Flagged $-$ plain", 1),
        ("written_gap", r"Written $-$ plain", 1),
        ("iclr_slope", "ICLR slope per year", 2),
        ("iclr_step", f"ICLR {last} $-$ {first}", 1),
    ]
    header = "Pipeline & Overlap & Cross-paper & " + " & ".join(title for _, title, _ in columns) + r" \\"
    lines = [r"\begin{tabular}{l" + "r" * (len(columns) + 2) + "}", r"\toprule", header, r"\midrule"]
    for r in contrasts.itertuples():
        cells = [
            f"{getattr(r, n):+.{d}f} [{getattr(r, n + '_low'):+.{d}f}, {getattr(r, n + '_high'):+.{d}f}]" for n, _, d in columns
        ]
        lines.append(
            f"{settings.MODEL_LABELS.get(r.pipeline, r.pipeline)} & {r.level:.1f} & {r.cross:.1f} & " + " & ".join(cells) + r" \\"
        )
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def unique_claim_group_ids(clusters):
    counts = Counter(cid for cluster in clusters for cid in cluster["claim_ids"])
    return {cid: group for cid, group in claim_group_ids(clusters).items() if counts[cid] == 1}


def council_pair_label(groupings, a, b):
    if len(groupings) != len(settings.GOLD_COUNCIL) or any(a not in g or b not in g for g in groupings):
        return "invalid", None
    votes = [g[a] == g[b] for g in groupings]
    return ("unanimous", votes[0]) if len(set(votes)) == 1 else ("split", None)


def council_agreement(
    model=settings.DEFAULT_MODEL,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "council.tex",
    council=settings.GOLD_COUNCIL,
):
    gold = gold_dir(gold)
    panels = list(read_jsonl(gold / "panels_tune.jsonl"))
    claims = extract.load_claims(gold / f"claims_{model}.jsonl")
    raw = {(row["panel_id"], row["model"]): row for row in read_jsonl(gold / "clusters_council.jsonl")}
    ours = load_jsonl(gold / f"clusters_{model}.jsonl", "panel_id")
    rows = []
    for panel in panels:
        pid, ids = panel["panel_id"], panel["review_ids"]
        if "clusters" not in ours.get(pid, {}):
            continue
        votes = []
        for member in council:
            proposal = raw.get((pid, member), {})
            if "clusters" in proposal:
                assert proposal["review_ids"] == ids, (pid, member)
            votes.append(unique_claim_group_ids(proposal.get("clusters", [])))
        model_groups = claim_group_ids(ours[pid]["clusters"])
        cids = [(i, f"R{i}-{k}") for i, rid in enumerate(ids) for k in range(len(claims[rid]))]
        for (ia, a), (ib, b) in combinations(cids, 2):
            if ia == ib:
                continue
            status, consensus = council_pair_label(votes, a, b)
            rows.append(
                dict(
                    **panel_meta(panel),
                    claim_a=a,
                    claim_b=b,
                    status=status,
                    consensus=consensus,
                    predicted=model_groups.get(a, a) == model_groups.get(b, b),
                    **{member: g[a] == g[b] if a in g and b in g else None for member, g in zip(council, votes)},
                )
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(Path(reports) / f"council_agreement_pairs_{model}.csv", index=False)
    n_reviews = {p["panel_id"]: len(p["review_ids"]) for p in panels}
    summary = []
    for kind, by_kind in pairs.groupby("kind", sort=False):
        for label, cell in report_cells(by_kind):
            unanimous = cell[cell.status == "unanimous"]
            counts = []
            for _, paper in unanimous.groupby("panel_id", sort=False):
                truth, predicted = paper.consensus.astype(bool), paper.predicted
                counts.append(
                    dict(
                        pair_tp=int((truth & predicted).sum()),
                        pair_fp=int((~truth & predicted).sum()),
                        pair_fn=int((truth & ~predicted).sum()),
                        pair_candidates=len(paper),
                    )
                )
            summary.append(
                dict(
                    kind=kind,
                    **label,
                    panels=cell.panel_id.nunique(),
                    reviews=sum(n_reviews[pid] for pid in set(cell.panel_id)),
                    candidate_pairs=len(cell),
                    invalid_pairs=int((cell.status == "invalid").sum()),
                    split_pairs=int((cell.status == "split").sum()),
                    unanimous_pairs=len(unanimous),
                    model_positives=int(cell.predicted.sum()),
                    retained_model_positives=int(unanimous.predicted.sum()),
                    **pair_detection_summary(pd.DataFrame(counts, columns=PAIR_COUNTS)),
                )
            )
    out = pd.DataFrame(summary)
    out.round(3).to_csv(Path(reports) / f"council_agreement_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    columns = [
        ("candidate_pairs", "Pairs", 0, False),
        ("unanimous_pairs", "Unanimous", 0, False),
        ("pair_precision", "Precision", 1, False),
        ("pair_recall", "Recall", 1, False),
        ("pair_f1", "F1", 1, False),
    ]
    write_cell_table(out, tex, columns)


QUOTE_CHUNK = 20000
FIELD_SPLIT = re.compile(r"(\[(?:" + "|".join(ai.SECTION_TAGS) + r")\])")
PRAISE_TAGS = {"[SUMMARY]", "[STRENGTHS]"}
MIN_SENTENCE_WORDS = 6
SHINGLE_WORDS = 5


def word_runs(text):
    words = extract.normalize_quote(text).split()
    return {n: {tuple(words[i : i + n]) for i in range(len(words) - n + 1)} for n in range(1, SHINGLE_WORDS + 1)}


def shares_run(a, b):
    n = min(max((n for n in a if a[n]), default=0), max((n for n in b if b[n]), default=0))
    return n > 0 and bool(a[n] & b[n])


def critique_sentences(text):
    pieces = FIELD_SPLIT.split(text)
    bodies = [pieces[0]] + [body for tag, body in zip(pieces[1::2], pieces[2::2]) if tag not in PRAISE_TAGS]
    sentences = [s for body in bodies for s in re.split(r"(?<=[.!?])\s+|\n+", body)]
    return [s for s in sentences if len(s.split()) >= MIN_SENTENCE_WORDS]


def quote_row(text, claims):
    quotes = [word_runs(c["quote"]) for c in claims]
    sentences = [word_runs(s) for s in critique_sentences(text)]
    return dict(
        claims=len(claims),
        found=sum(extract.quote_in_text(c["quote"], text) for c in claims),
        duplicated=sum(any(shares_run(q, other) for other in quotes[:i] + quotes[i + 1 :]) for i, q in enumerate(quotes)),
        sentences=len(sentences),
        uncovered=sum(not any(shares_run(s, q) for q in quotes) for s in sentences),
    )


def quotes(model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR):
    claims = hub.claims(model)
    ids = list(claims)
    rows = []
    for start in range(0, len(ids), QUOTE_CHUNK):
        chunk = ids[start : start + QUOTE_CHUNK]
        texts = hub.texts(chunk)
        rows += [dict(review_id=rid, **quote_row(texts[rid], claims[rid])) for rid in chunk]
    df = pd.DataFrame(rows).join(hub.meld_flags(["conf", "year", "flag"]), on="review_id")
    counts = ["claims", "found", "duplicated", "sentences", "uncovered"]
    out = df.groupby(["conf", "year", "flag"], dropna=False)[counts].sum().reset_index()
    out["found_pct"] = 100 * out["found"] / out["claims"]
    out["duplicated_pct"] = 100 * out["duplicated"] / out["claims"]
    out["uncovered_pct"] = 100 * out["uncovered"] / out["sentences"]
    out.to_csv(Path(reports) / f"quotes_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def item_alignment(items, claims):
    item_runs = [word_runs(item) for item in items]
    hits = [[i for i, item in enumerate(item_runs) if shares_run(item, word_runs(c["quote"]))] for c in claims]
    counts = Counter(i for hit in hits for i in hit)
    return dict(
        items=len(items),
        claims=len(claims),
        items_hit=len(counts),
        items_split=sum(n > 1 for n in counts.values()),
        claims_unmatched=sum(not hit for hit in hits),
        claims_spanning=sum(len(hit) > 1 for hit in hits),
    )


def external(name, model=settings.DEFAULT_MODEL, gold=settings.GOLD_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    gold = gold_dir(gold)
    items = {row["review_id"]: row["items"] for row in read_jsonl(gold / f"{name}_items.jsonl")}
    claims = extract.load_claims(gold / f"claims_{model}_{name}.jsonl")
    df = pd.DataFrame([dict(review_id=rid, **item_alignment(items[rid], claims[rid])) for rid in items if rid in claims])
    out = pd.DataFrame([external_summary(df)])
    out.round(3).to_csv(Path(reports) / f"{name}_{model}.csv", index=False)
    print(out.round(2).T.to_string(header=False))


def external_summary(df):
    summary = dict(reviews=len(df), items=df["items"].sum(), claims=df["claims"].sum())
    shares = [
        ("item_recall", "items_hit", "items"),
        ("item_split", "items_split", "items"),
        ("claim_unmatched", "claims_unmatched", "claims"),
        ("claim_spanning", "claims_spanning", "claims"),
    ]
    for name, numerator, denominator in shares:
        summary.update(mean_interval(100 * df[numerator] / df[denominator], name))
    summary.update(mean_interval(df["claims"] - df["items"], "claims_bias"))
    return summary


if __name__ == "__main__":
    fire.Fire(
        {
            "robust": robust,
            "forms": forms,
            "detector_placebo": detector_placebo,
            "rq1_evidence": rq1_evidence,
            "journal_gaps": journal_gaps,
            "extraction": extraction,
            "council_agreement": council_agreement,
            "pipelines": pipelines,
            "identification": identification,
            "pair_types": pair_types,
            "pair_type_sizes": pair_type_sizes,
            "detector_choices": detector_choices,
            "journal_subsets": journal_subsets,
            "coverage": coverage,
            "rewrite_test": rewrite_test,
            "repro": repro,
            "quotes": quotes,
            "external": external,
        }
    )
