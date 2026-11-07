import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm

import hub
from ai import PAIR_PREFIX, REWRITE_CONDITIONS
from extract import read_jsonl
from meld import WRITTEN_TASK
from metrics import assign, load_jsonl, mean_overlap, review_groups
from metrics import overlap as pair_overlap
from outcomes import ICLR_FROM, ICLR_LAST, excluded_review_ids, in_window
from settings import settings

ML_VENUES = ["ICLR", "NeurIPS", "ICML"]
REMOVED_FIELD_TAGS = ("_nocq", "_noq")
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
    checks = [
        ("Main overlap rate", main, "hit_rate_strict", 2023, ICLR_LAST),
        ("Specific claims only", specific, "hit_spec", 2023, ICLR_LAST),
        ("Same part of paper (lenient)", main, "hit_rate_lenient", 2023, ICLR_LAST),
        ("3-review subsets, grouped again", sub3, "hit_rate_strict", 2023, ICLR_LAST),
        ("Questions field removed", noq, "hit_rate_strict", 2024, ICLR_LAST),
        ("Withdrawn 2026 papers added", with_wd, "hit_rate_strict", 2023, 2026),
    ]
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
    lines += [
        f"{r.check} & {r.base_year} & {r.base:.1f} & {r.last_year} & {r.last:.1f} & {r.change:+.1f} ({r.se:.1f}) \\\\"
        for r in out.itertuples()
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


def meld_flags():
    return hub.meld_flags(["review_id", "year", "score", "flag", "written"]).set_index("review_id")


def meld_flag_at(meld, quantile):
    threshold = meld[meld.year <= settings.MELD_CALIBRATION_LAST_YEAR].score.quantile(quantile)
    return (meld.score > threshold).astype(int)


def detector_placebo(model=settings.DEFAULT_MODEL, share=0.45, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    iclr_pairs = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    iclr_pairs = iclr_pairs[iclr_pairs.conf == "ICLR"]
    scores = meld_flags()
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
    meld = meld_flags()
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
    flag = meld_flags().flag.astype(int)
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
    window = pairs[pairs.conf.isin(ML_VENUES) & (pairs.year >= 2024)]
    return list(window.groupby("conf"))


def pair_types(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    meld = meld_flags()
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
    meld = meld_flags()
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
    flag = flags.drop_duplicates("review_id").set_index("review_id").flag.astype(int)
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
    meld = hub.meld_flags(["review_id", "conf", "year", "score", "task"])
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    pairs = pairs[(pairs.conf == "ICLR") & (pairs.year >= 2024)]
    iclr_last = meld[(meld.conf == "ICLR") & (meld.year == pairs.year.max())].review_id
    iclr_last = iclr_last[~iclr_last.isin(excluded_review_ids())]
    calibration = meld[meld.year <= settings.MELD_CALIBRATION_LAST_YEAR].score
    rows = []
    for quantile in DETECTOR_QUANTILES:
        flagged = meld.assign(flag=meld.score > calibration.quantile(quantile))
        written = flagged.assign(flag=flagged.flag & (flagged.task == WRITTEN_TASK))
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
    flag = meld_flags().flag.astype(int)
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    journals = pairs[~pairs.conf.isin(ML_VENUES) & (pairs.year >= 2024)]
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
    meld = meld_flags()
    flag, written = meld.flag.astype(int), meld.written.astype(int)
    pairs = in_window(
        pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    )
    iclr = pairs[(pairs.conf == "ICLR") & (pairs.year >= 2024)]
    journals = pairs[~pairs.conf.isin(ML_VENUES) & (pairs.year >= 2024)]
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
        f"conf == 'ICLR' and year >= {ICLR_FROM}"
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
    flags = hub.meld_flags(["review_id", "conf", "year"])
    calibration = flags[flags.year <= settings.MELD_CALIBRATION_LAST_YEAR]
    in_analysis = calibration.review_id.isin(set(pairs.a) | set(pairs.b))
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
            clusters.get(f"{PAIR_PREFIX}rw0_{p['panel_id']}"),
            clusters.get(f"{PAIR_PREFIX}{panel_prefix}{p['panel_id']}"),
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
    matcher=settings.GOLD_JUDGE_MODEL,
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
    for condition, c in REWRITE_CONDITIONS.items():
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


def grouped(review_ids, claims, clusters):
    group_of = {}
    for g, cluster in enumerate(clusters):
        for cid in cluster["claim_ids"]:
            group_of.setdefault(cid, g)
    return [{group_of.get(f"R{j}-{k}", f"R{j}-{k}") for k in range(len(claims[rid]))} for j, rid in enumerate(review_ids)]


def strict_overlap(review_ids, claims, clusters):
    return mean_overlap(grouped(review_ids, claims, clusters))


def audit_pairs(gold, model_clusters, model_claims, flag, written, stratum_of):
    ref_claims = {rid: row["claims"] for rid, row in load_jsonl(gold / "adjudicate" / "out.jsonl", "review_id").items()}
    ref_clusters = load_jsonl(gold / "adjudicate2" / "out.jsonl", "panel_id")
    rows = []
    for panel in read_jsonl(gold / "panels_tune.jsonl"):
        if panel["kind"] != "same":
            continue
        ids, ours = panel["review_ids"], model_clusters[panel["panel_id"]]
        assert ours["review_ids"] == ids, panel["panel_id"]
        ref = grouped(ids, ref_claims, ref_clusters[panel["panel_id"]]["clusters"])
        luna = grouped(ids, model_claims, ours["clusters"])
        for i, j in combinations(range(len(ids)), 2):
            rows.append(
                dict(
                    panel=f"{gold.name}:{panel['panel_id']}",
                    stratum=stratum_of(panel),
                    year=panel["year"],
                    n_flagged=int(flag.get(ids[i], 0)) + int(flag.get(ids[j], 0)),
                    n_written=int(written.get(ids[i], 0)) + int(written.get(ids[j], 0)),
                    ref=100 * pair_overlap(ref[i], ref[j]),
                    luna=100 * pair_overlap(luna[i], luna[j]),
                )
            )
    return pd.DataFrame(rows)


AUDIT_STRATA = {
    "same2023": "ICLR 2023",
    "flag": "ICLR 2024--26, flag strata",
    "written": "ICLR 2024--26, LLM-written strata",
    "journal": "Journals 2024--26",
}


def type_gap(pairs, label, pick):
    picked = pairs.assign(kind=pick(pairs)).dropna(subset=["kind"])
    by_type = picked.groupby(["panel", "kind"])[["ref", "luna", "error"]].mean().unstack()
    both = by_type.dropna(subset=[("ref", 0), ("ref", 2)])
    gap = {name: both[(name, 2)] - both[(name, 0)] for name in ("ref", "luna", "error")}
    return dict(
        row="gap",
        label=label,
        panels=len(both),
        ref=gap["ref"].mean(),
        luna=gap["luna"].mean(),
        **mean_interval(gap["error"], "diff"),
        **mean_interval(gap["ref"], "ref_gap"),
        **mean_interval(gap["luna"], "luna_gap"),
    )


def audit(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    gold=settings.GOLD_DIR,
    audit_dir=settings.AUDIT_DIR,
    written_dir=settings.AUDIT_WRITTEN_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "audit.tex",
):
    data, gold, audit_dir, written_dir = Path(data), Path(gold), Path(audit_dir), Path(written_dir)
    labels = meld_flags()
    flag, written = labels.flag.to_dict(), labels.written.to_dict()
    tunes = {d: list(read_jsonl(d / "panels_tune.jsonl")) for d in [gold, audit_dir, written_dir]}
    model_claims = hub.claims(model, {r for panels in tunes.values() for panel in panels for r in panel["review_ids"]})
    main_clusters = hub.clusters(model, panel_ids=[panel["panel_id"] for panel in tunes[gold]])
    audit_clusters = load_jsonl(audit_dir / f"clusters_{model}.jsonl", "panel_id")
    written_clusters = load_jsonl(written_dir / f"clusters_{model}.jsonl", "panel_id")
    pairs = pd.concat(
        [
            audit_pairs(gold, main_clusters, model_claims, flag, written, lambda p: f"ICLR {p['year']}"),
            audit_pairs(audit_dir, audit_clusters, model_claims, flag, written, lambda p: AUDIT_STRATA[p["stratum"]]),
            audit_pairs(written_dir, written_clusters, model_claims, flag, written, lambda p: AUDIT_STRATA[p["stratum"]]),
        ]
    )
    pairs["error"] = pairs.luna - pairs.ref
    pairs.to_parquet(settings.ANALYSIS_DIR / f"audit_pairs_{model}.parquet")
    panels = pairs.groupby(["stratum", "panel"])[["ref", "luna", "error"]].mean().reset_index()
    order = ["ICLR 2018", "ICLR 2022", "ICLR 2023", "ICLR 2024", "ICLR 2026", *list(AUDIT_STRATA.values())[1:]]
    rows = []
    for stratum in order:
        cell = panels[panels.stratum == stratum]
        rows.append(
            dict(
                row="level",
                label=stratum,
                panels=len(cell),
                ref=cell.ref.mean(),
                luna=cell.luna.mean(),
                **mean_interval(cell.error, "diff"),
            )
        )
    rows.append(type_gap(pairs, "Two flagged $-$ none", lambda d: d.n_flagged.where(d.n_flagged != 1)))
    rows.append(
        type_gap(
            pairs, "Two LLM-written $-$ none", lambda d: d.n_written.where(d.n_written == 2, d.n_flagged.where(d.n_flagged == 0))
        )
    )
    early = panels[panels.stratum.isin(["ICLR 2018", "ICLR 2022", "ICLR 2023"])]
    late = panels[panels.stratum.isin(["ICLR 2024", "ICLR 2026"])]
    rows.append(
        dict(
            row="change",
            label="ICLR 2024--26 minus 2018--23",
            panels=len(early) + len(late),
            **welch_interval(late.error, early.error, "diff"),
        )
    )
    out = pd.DataFrame(rows)
    out.round(3).to_csv(Path(reports) / f"audit_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Panels & $n$ & Ref. & Luna & Luna $-$ ref. [95\% CI] \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        ref = "--" if pd.isna(r.ref) else f"{r.ref:.1f}"
        luna = "--" if pd.isna(r.luna) else f"{r.luna:.1f}"
        if r.row != "level" and out.row.iloc[r.Index - 1] == "level":
            lines.append(r"\midrule")
        lines.append(f"{r.label} & {r.panels} & {ref} & {luna} & {r.diff:+.1f} [{r.diff_low:+.1f}, {r.diff_high:+.1f}] \\\\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


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
    tune = list(read_jsonl(gold / "panels_tune.jsonl"))
    model_clusters = hub.clusters(model, panel_ids=[panel["panel_id"] for panel in tune])
    model_claims = hub.panel_claims(model, model_clusters)
    rows = []
    for panel in tune:
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
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r" & & & Claims & \multicolumn{3}{c}{Overlap (\%)} & \multicolumn{2}{c}{Matched (\%)} \\",
        r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"Panels & Years & $n$ & Diff. (SE) & Ref. & Luna & Diff. (SE) & Recall & Precision \\",
        r"\midrule",
    ]
    names = {"same": "Same-paper", "cross": "Cross-paper"}
    for r in out[out.split == "all"].itertuples():
        years = r.years.replace(" minus earlier", r" $-$ earlier").replace("all", "All")
        if "earlier" in years:
            cells = f"{r.claims_bias:+.2f} ({r.claims_bias_se:.2f}) & & & {r.overlap_bias:+.1f} ({r.overlap_bias_se:.1f})"
            matched = f"{r.recall:+.1f} ({r.recall_se:.1f}) & {r.precision:+.1f} ({r.precision_se:.1f})"
        else:
            cells = (
                f"{r.claims_bias:+.2f} ({r.claims_bias_se:.2f}) & "
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
            "identification": identification,
            "pair_types": pair_types,
            "pair_type_sizes": pair_type_sizes,
            "detector_choices": detector_choices,
            "journal_subsets": journal_subsets,
            "coverage": coverage,
            "rewrite_test": rewrite_test,
            "audit": audit,
            "repro": repro,
        }
    )
