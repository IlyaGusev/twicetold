import re
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

import ai
import extract
import hub
from between import normalize_rows, openai_embed
from jsonl import load_jsonl, read_jsonl
from metrics import assign, mean_overlap, review_groups
from metrics import overlap as pair_overlap
from outcomes import analysis_pairs, in_window
from settings import settings

REMOVED_FIELD_TAGS = ("_nocq", "_noq")
HALF_SPAN = 0.5
LABEL_FIELDS = ("category", "target", "specific")
PAIR_COUNTS = ["pair_tp", "pair_fp", "pair_fn", "pair_candidates"]
COVER_QUANTILE = 0.95
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
    first, last = settings.ICLR_SNAPSHOT_YEARS
    status = hub.corpus_reviews(["paper_id", "conf", "year", "withdrawn", "decision"]).query("conf == 'ICLR' and year >= @first")
    status = status.groupby(["year", "paper_id"]).agg(
        reviews=("paper_id", "size"), withdrawn=("withdrawn", "first"), decision=("decision", "first")
    )
    status = status[status.reviews >= settings.MIN_PANEL_REVIEWS]
    status["left"] = status.withdrawn | status.decision.eq("desk_reject")
    left_share = status.left.groupby("year").mean()
    left_ids = status.index.get_level_values("paper_id")[status.left]
    kept, left = main[~main.paper_id.isin(left_ids)], main[main.paper_id.isin(left_ids)]

    def level(df, col, year):
        values = df[df.year == year][col].dropna() * 100
        return values.mean(), values.sem()

    left_level = level(left, "hit_rate_strict", first)
    pd.DataFrame(
        [
            dict(name="left_share", value=100 * left_share[first]),
            dict(name="last_left_share", value=100 * left_share[last]),
            dict(name="left_papers", value=left[left.year == first].hit_rate_strict.notna().sum()),
            dict(name="left_level", value=left_level[0]),
            dict(name="left_level_se", value=left_level[1]),
        ]
    ).to_csv(Path(reports) / f"withdrawn_{model}.csv", index=False)
    variants = [
        ("Overlap", main, "hit_rate_strict"),
        (settings.WITHDRAWN_CHECK, kept, "hit_rate_strict"),
        ("Specific claims only", specific, "hit_spec"),
        ("Same part of paper (lenient)", main, "hit_rate_lenient"),
        ("3-review subsets, grouped again", sub3, "hit_rate_strict"),
        ("Questions field removed", noq, "hit_rate_strict"),
    ]
    checks = [(name, level(df, col, first), level(df, col, last), first, last) for name, df, col in variants]
    base_year, last_year = settings.CONTRAST_YEARS
    checks += [(name, level(df, col, base_year), level(df, col, last_year), base_year, last_year) for name, df, col in variants]
    rows = []
    for name, base, end, base_year, last_year in checks:
        rows.append(
            {
                "check": name,
                "base_year": base_year,
                "base": base[0],
                "last_year": last_year,
                "last": end[0],
                "change": end[0] - base[0],
                "se": np.hypot(base[1], end[1]),
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
        if k and r.last_year != out.last_year.iloc[k - 1]:
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


FORM_COLUMNS = ["hit", "hit_removed", "hit_without", "form_share"]


def forms_table(out, tex):
    lines = [
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{4.3cm}rrrrr@{}}",
        r"\toprule",
        r"Year & Text fields & Papers & \shortstack[r]{Overlap\\(\%)} & \shortstack[r]{Overlap, field\\deleted (\%)} & "
        r"\shortstack[r]{Overlap, categories\\dropped (\%)} & \shortstack[r]{Shared problems\\in categories (\%)} \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        fields = sorted(r.form.split("+"), key=list(FIELD_LABELS).index)
        cells = ["--" if pd.isna(getattr(r, c)) else f"{getattr(r, c):.1f}" for c in FORM_COLUMNS]
        lines.append(f"{r.year} & {'; '.join(FIELD_LABELS[f] for f in fields)} & {r.papers:,} & {' & '.join(cells)} \\\\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def forms_table_from_report(
    model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR, tex=settings.TABLES_DIR / "forms.tex"
):
    forms_table(pd.read_csv(Path(reports) / f"forms_{model}.csv"), tex)


def forms(
    model=settings.DEFAULT_MODEL,
    data=settings.WORK_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "forms.tex",
):
    data = Path(data)
    main = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet").query(
        f"conf == 'ICLR' and year >= {settings.ICLR_FROM} and n_claims > 0"
    )
    clusters = hub.clusters(model, panel_ids=main.panel_id)
    claims = hub.panel_claims(model, clusters, ["category"])
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
    forms_table(out, tex)


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


def paper_gap(design, groups, top, base):
    means = design.hit.groupby([design.panel_id.to_numpy(), np.asarray(groups)]).mean().unstack()
    if not {top, base} <= set(means.columns):
        return dict(est=np.nan, se=np.nan, papers=0)
    difference = (means[top] - means[base]).dropna() * 100
    return dict(est=difference.mean(), se=difference.sem(), papers=len(difference))


def direct_gap(design, flag):
    counts = pair_counts(design, flag).to_numpy()
    two = paper_gap(design, counts, 2, 0)
    return {"two": two["est"], "two_se": two["se"], "papers": two["papers"]}


def meld_flag_at(meld, quantile):
    threshold = meld[meld.year <= settings.MELD_CALIBRATION_LAST_YEAR].score.quantile(quantile)
    return (meld.score > threshold).astype(int)


def detector_placebo(model=settings.DEFAULT_MODEL, share=0.45, data=settings.WORK_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    iclr_pairs = analysis_pairs(model)
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
        gap = direct_gap(period_pairs, flag)
        papers = period_pairs.panel_id.nunique()
        rows.append(dict(period=name, two=gap["two"], two_se=gap["two_se"], papers=papers, identifying=gap["papers"]))
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / f"detector_placebo_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def short_long_logs(values, design):
    log_a = np.log(values.reindex(design.a).to_numpy())
    log_b = np.log(values.reindex(design.b).to_numpy())
    return np.minimum(log_a, log_b), np.maximum(log_a, log_b)


def pair_design(model):
    design = analysis_pairs(model)
    chars = hub.text_chars(set(design.a) | set(design.b))
    design["log_short"], design["log_long"] = short_long_logs(chars, design)
    return design


def rq1_evidence(model=settings.DEFAULT_MODEL, data=settings.WORK_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    data, reports = Path(data), Path(reports)
    meld = hub.meld_flags()
    flag = meld.flag.astype(int)
    design = pair_design(model)
    claim_counts = pd.read_parquet(data / f"claims_table_{model}.parquet", columns=["review_id"]).groupby("review_id").size()
    design["claims_short"], design["claims_long"] = short_long_logs(claim_counts, design)
    cells = gap_cells(design)
    periods = iclr_cells(design)
    rows = []

    def add(group, label, subset_pairs, pair_flags, controls=()):
        if controls:
            gap = fe_gap(subset_pairs, pair_flags, controls)
            n_flagged = pair_flags.reindex(subset_pairs.a).to_numpy() + pair_flags.reindex(subset_pairs.b).to_numpy()
            compared = pd.Series(n_flagged).groupby(subset_pairs.panel_id.to_numpy()).nunique().gt(1).sum()
        else:
            gap = direct_gap(subset_pairs, pair_flags)
            compared = gap["papers"]
        rows.append(
            dict(
                group=group,
                label=label,
                est=gap["two"],
                se=gap["two_se"],
                papers=subset_pairs.panel_id.nunique(),
                identifying=compared,
            )
        )

    for period, iclr in periods.items():
        add(period, period, iclr, flag)
        controls = ("log_short", "log_long", "claims_short", "claims_long")
        add(period, f"{period}, + length and claim count", iclr.dropna(), flag, controls)
        add(period, f"{period}, 99th-percentile threshold", iclr, meld_flag_at(meld, 0.99))
        add(period, f"{period}, 99.5th-percentile threshold", iclr, meld_flag_at(meld, 0.995))
        for year, year_pairs in iclr.groupby("year"):
            add(period, f"ICLR {year}", year_pairs, flag)
    for label, cell in cells.items():
        if label not in periods:
            add("Other venues", label, cell, flag)
    placebo = pd.read_csv(reports / f"detector_placebo_{model}.csv")
    for r in placebo.itertuples():
        rows.append(
            dict(
                group="Placebo before ChatGPT",
                label=f"Top-45% scores, ICLR {r.period}",
                est=r.two,
                se=r.two_se,
                papers=r.papers,
                identifying=r.identifying,
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"rq1_evidence_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def journal_gaps(model=settings.DEFAULT_MODEL, data=settings.WORK_DIR, reports=settings.ANALYSIS_REPORTS_DIR, first_year=2024):
    flag = hub.meld_flags().flag.astype(int)
    pairs = analysis_pairs(model)
    journals = pairs[~pairs.conf.isin(settings.ML_VENUES) & (pairs.year >= first_year)]
    rows = []
    for source, source_pairs in [*journals.groupby("conf"), ("All journals", journals)]:
        gap = fe_gap(source_pairs, flag)
        types = pair_counts(source_pairs, flag).groupby(source_pairs.panel_id.to_numpy()).nunique()
        rows.append(
            dict(
                source=source,
                est=gap["two"],
                se=gap["two_se"],
                papers=source_pairs.panel_id.nunique(),
                varying=int((types >= 2).sum()),
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


PAIR_TYPES = ("HL", "LL")


def pair_type(pairs, flag):
    return pair_counts(pairs, flag).map({0: "HH", 1: "HL", 2: "LL"})


def gap_cells(pairs):
    return {
        label: pairs[(pairs.conf == conf) & pairs.year.between(first, last)]
        for label, (conf, first, last) in settings.GAP_CELLS.items()
    }


def iclr_cells(pairs):
    return {label: cell for label, cell in gap_cells(pairs).items() if settings.GAP_CELLS[label][0] == "ICLR"}


def pair_types(model=settings.DEFAULT_MODEL, data=settings.WORK_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    flag = hub.meld_flags().flag.astype(int)
    pairs = analysis_pairs(model)
    rows = []
    for conf, cell in gap_cells(pairs).items():
        types = pair_type(cell, flag).to_numpy()
        X = pd.DataFrame({t: (types == t) * 1.0 for t in PAIR_TYPES} | {"hit": cell.hit.to_numpy() * 100}, index=cell.index)
        X = X - X.groupby(cell.panel_id.to_numpy()).transform("mean")
        fit = sm.OLS(X.hit, X[list(PAIR_TYPES)]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(cell.panel_id)[0]})
        for t in PAIR_TYPES:
            rows.append(dict(conf=conf, pair_type=t, **paper_gap(cell, types, t, "HH"), est_fe=fit.params[t], se_fe=fit.bse[t]))
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / f"pair_types_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


PAIR_SIZE_OUTCOMES = {"hit": "Overlap", "shared": "Shared", "smaller": "Smaller set", "union": "Union"}
PAIR_TYPE_NAMES = {
    "HH": "Both unflagged",
    "HL": "One flagged",
    "LL": "Both flagged",
}


def type_fit(cell, types, outcome, controls=()):
    X = pd.DataFrame({t: (types == t) * 1.0 for t in PAIR_TYPES} | {"y": cell[outcome].to_numpy()}, index=cell.index)
    X = X.join(cell[list(controls)])
    X = X - X.groupby(cell.panel_id.to_numpy()).transform("mean")
    return sm.OLS(X.y, X[[*PAIR_TYPES, *controls]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(cell.panel_id)[0]})


def pair_type_sizes_table(out, tex):
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Pair type & Pairs & \shortstack[r]{Overlap\\(points)} & \shortstack[r]{Shared\\problems} & "
        r"\shortstack[r]{Problems,\\smaller review} & \shortstack[r]{Problems,\\either review} & "
        r"\shortstack[r]{Shared,\\sizes fixed} \\",
    ]
    for r in out.itertuples():
        if r.pair_type == "HH":
            lines += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{r.cell.replace('–', '--')}}}}} \\"]
            cells = [f"{r.hit:.1f}", f"{r.shared:.2f}", f"{r.smaller:.2f}", f"{r.union:.2f}", f"{r.shared:.2f}"]
        else:
            cells = [f"{getattr(r, name):+.{1 if name == 'hit' else 2}f}" for name in [*PAIR_SIZE_OUTCOMES, "shared_adjusted"]]
        lines.append(f"{PAIR_TYPE_NAMES[r.pair_type]} & {r.pairs:,} & " + " & ".join(cells) + r" \\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def pair_type_sizes_table_from_report(
    model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR, tex=settings.TABLES_DIR / "pair_type_sizes.tex"
):
    pair_type_sizes_table(pd.read_csv(Path(reports) / f"pair_type_sizes_{model}.csv"), tex)


def pair_type_sizes(
    model=settings.DEFAULT_MODEL,
    data=settings.WORK_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "pair_type_sizes.tex",
):
    flag = hub.meld_flags().flag.astype(int)
    pairs = analysis_pairs(model, None)
    rows = []
    for period, cell in iclr_cells(pairs).items():
        cell = cell.assign(hit=cell.hit * 100)
        types = pair_type(cell, flag).to_numpy()
        fits = {name: type_fit(cell, types, name) for name in PAIR_SIZE_OUTCOMES}
        fits["shared_adjusted"] = type_fit(cell, types, "shared", ("smaller", "larger"))
        for t in ("HH", *PAIR_TYPES):
            row = dict(cell=period, pair_type=t, pairs=int((types == t).sum()))
            for name, fit in fits.items():
                if t == "HH":
                    row[name] = cell[types == "HH"][name.replace("_adjusted", "")].mean()
                else:
                    row[name], row[name + "_se"] = fit.params[t], fit.bse[t]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / f"pair_type_sizes_{model}.csv", index=False)
    print(out.round(3).to_string(index=False))
    pair_type_sizes_table(out, tex)


JOURNAL_SUBSETS = ["Identifying", "3--5 reports, other", "Two reports"]
JOURNAL_ORDER = ["F1000", "PLOS", "Copernicus", "BMC-Med", "SciPost"]


def journal_subsets(
    model=settings.DEFAULT_MODEL,
    data=settings.WORK_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "journal_subsets.tex",
):
    data = Path(data)
    flag = hub.meld_flags().flag.astype(int)
    pairs = analysis_pairs(model)
    journals = pairs[~pairs.conf.isin(settings.ML_VENUES) & (pairs.year >= 2024)]
    journals = journals.assign(n_flagged=pair_counts(journals, flag).to_numpy())
    texts = pd.concat(
        [journals[["panel_id", "a"]].rename(columns={"a": "r"}), journals[["panel_id", "b"]].rename(columns={"b": "r"})]
    ).drop_duplicates()
    chars = hub.text_chars(set(texts.r))
    wanted = set(texts.r)
    claims = pd.Series({r: len(c) for r, c in hub.claims(model, wanted, ["specific"]).items()})
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
    data=settings.WORK_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
):
    flag = hub.meld_flags().flag.astype(int)
    pairs = analysis_pairs(model)
    journals = pairs[~pairs.conf.isin(settings.ML_VENUES) & (pairs.year >= 2024)]
    cells = [
        *((label, cell, flag) for label, cell in iclr_cells(pairs).items()),
        ("NeurIPS 2024–25", pairs[(pairs.conf == "NeurIPS") & (pairs.year >= 2024)], flag),
        ("ICML 2025–26", pairs[pairs.conf == "ICML"], flag),
        *[(f"{source} {cell.year.min()}–{cell.year.max() % 100}", cell, flag) for source, cell in journals.groupby("conf")],
        ("All journals 2024–26", journals, flag),
    ]
    out = pd.DataFrame([identification_row(label, cell, f) for label, cell, f in cells])
    out.round(3).to_csv(Path(reports) / f"identification_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


ABSTRACT_QUANTILES = (0.999, 0.99)
ABSTRACT_TERMS = ("change", "composition", "within_flagged", "within_other")


def paper_outcomes(model):
    columns = ["panel_id", "conf", "year", "n_reviews", "hit_rate_strict"]
    ml = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet", columns=[*columns, "paper_id"])
    icml = pd.read_parquet(settings.ANALYSIS_DIR / f"icml_panel_outcomes_{model}.parquet", columns=columns)
    icml = icml.assign(paper_id=icml.panel_id.str.removeprefix("icml_"))
    papers = in_window(pd.concat([ml, icml], ignore_index=True)).dropna(subset=["hit_rate_strict"])
    return papers.assign(overlap=papers.hit_rate_strict * 100)


def flagged_review_share(model):
    reviews = pd.read_parquet(settings.ANALYSIS_DIR / f"review_groups_{model}.parquet", columns=["panel_id", "review_id"])
    flags = hub.detector_flags("editlens").astype(float)
    return reviews.review_id.map(flags).groupby(reviews.panel_id.to_numpy()).mean().mul(100).rename("flagged_reviews")


def flag_means(frame):
    flagged, other = frame.overlap[frame.flag], frame.overlap[~frame.flag]
    se = np.sqrt(flagged.var() / len(flagged) + other.var() / len(other)) if len(flagged) > 1 else np.nan
    return dict(
        papers=len(frame),
        flagged_papers=int(frame.flag.sum()),
        flagged_share=frame.flag.mean() * 100,
        overlap_flagged=flagged.mean(),
        overlap_other=other.mean(),
        est=flagged.mean() - other.mean(),
        se=se,
    )


def flag_fit(frame, controls=()):
    X = pd.get_dummies(frame.year, prefix="year", dtype=float).assign(flag=frame.flag.astype(float))
    X = X.join(frame[list(controls)])
    fit = sm.OLS(frame.overlap, X).fit(cov_type="HC1")
    return dict(papers=len(frame), flagged_papers=int(frame.flag.sum()), est=fit.params["flag"], se=fit.bse["flag"])


def abstract_terms(f0, o0, f1, o1):
    s0, s1 = f0.mean(), f1.mean()
    a0, a1 = (o[f].mean() if f.any() else np.nan for f, o in ((f0, o0), (f1, o1)))
    b0, b1 = o0[~f0].mean(), o1[~f1].mean()
    a0, a1 = (a1 if np.isnan(a0) else a0), (a0 if np.isnan(a1) else a1)
    return np.array(
        [
            o1.mean() - o0.mean(),
            (s1 - s0) * ((a0 - b0) + (a1 - b1)) / 2,
            (a1 - a0) * (s0 + s1) / 2,
            (b1 - b0) * (1 - (s0 + s1) / 2),
        ]
    )


def abstract_decomposition(before, after, rng):
    (f0, o0), (f1, o1) = ((cell.flag.to_numpy(), cell.overlap.to_numpy()) for cell in (before, after))
    terms = abstract_terms(f0, o0, f1, o1)
    draws = []
    for _ in range(settings.REFERENCE_BOOTSTRAPS):
        i0, i1 = rng.integers(len(f0), size=len(f0)), rng.integers(len(f1), size=len(f1))
        draws.append(abstract_terms(f0[i0], o0[i0], f1[i1], o1[i1]))
    errors = np.std(draws, axis=0)
    row = dict(zip(ABSTRACT_TERMS, terms, strict=True)) | {f"{t}_se": e for t, e in zip(ABSTRACT_TERMS, errors, strict=True)}
    return row | dict(
        share_before=before.flag.mean() * 100, share_after=after.flag.mean() * 100, explained=100 * terms[1] / terms[0]
    )


def abstract_papers(model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR):
    outcomes = paper_outcomes(model)
    outcomes = outcomes.join(flagged_review_share(model), on="panel_id")
    rng = np.random.default_rng(settings.REFERENCE_BOOTSTRAP_SEED)
    gaps, steps = [], []
    for detector in settings.ABSTRACT_DETECTORS:
        scores = hub.abstract_scores(detector)
        calibration = scores[scores.year <= settings.MELD_CALIBRATION_LAST_YEAR].score
        papers = outcomes.merge(scores[["conf", "paper_id", "score"]], on=["conf", "paper_id"])
        for quantile in ABSTRACT_QUANTILES:
            cut = dict(detector=detector, percentile=quantile * 100)
            papers = papers.assign(flag=papers.score > calibration.quantile(quantile))
            for (conf, year), cell in papers[papers.year >= settings.MELD_CALIBRATION_LAST_YEAR + 1].groupby(["conf", "year"]):
                gaps.append(cut | dict(group="By year", label=f"{conf} {year}") | flag_means(cell))
            for label, (conf, first, last) in settings.GAP_CELLS.items():
                cell = papers[(papers.conf == conf) & papers.year.between(first, last)]
                gaps.append(cut | dict(group="Year effects", label=label) | flag_fit(cell))
                controlled = cell.dropna(subset=["flagged_reviews"])
                gaps.append(
                    cut
                    | dict(group="Year effects", label=f"{label}, + flagged-review share and review count")
                    | flag_fit(controlled, ("flagged_reviews", "n_reviews"))
                )
                for year in range(first, last):
                    before, after = (papers[(papers.conf == conf) & (papers.year == y)] for y in (year, year + 1))
                    steps.append(cut | dict(conf=conf, year=year + 1) | abstract_decomposition(before, after, rng))
            last = papers[(papers.conf == "ICLR") & (papers.year == settings.ICLR_LAST)].flag.mean()
            pre = papers[(papers.conf == "ICLR") & papers.year.between(2020, settings.MELD_CALIBRATION_LAST_YEAR)]
            pre = pre.assign(flag=pre.score >= pre.groupby("year").score.transform("quantile", 1 - last))
            label = f"Top {last * 100:.0f}% scores, ICLR 2020-{settings.MELD_CALIBRATION_LAST_YEAR}"
            gaps.append(cut | dict(group="Placebo before ChatGPT", label=label) | flag_fit(pre))
    gaps, steps = pd.DataFrame(gaps), pd.DataFrame(steps).drop_duplicates(["detector", "percentile", "conf", "year"])
    gaps.to_csv(Path(reports) / f"abstract_gaps_{model}.csv", index=False)
    steps.to_csv(Path(reports) / f"abstract_decomposition_{model}.csv", index=False)
    print(gaps.round(2).to_string(index=False))
    print(steps.round(2).to_string(index=False))


APPENDIX_PATTERN = re.compile(
    r"\bappendi(?:x|ces)\b|\bsupplementa(?:ry|l)\b|\bapp\.\s?[A-H]\b"
    r"|\b(?:table|figure|fig\.|section|sec\.|equation|eq\.|theorem|lemma|algorithm)\s?\(?[A-H]\.?\d",
    re.IGNORECASE,
)


def appendix_marks(claims):
    texts = pd.Series([f"{c['quote'] or ''} {c['claim'] or ''}" for items in claims.values() for c in items])
    marks = iter(texts.str.contains(APPENDIX_PATTERN).tolist())
    return {review_id: [dict(appendix=next(marks)) for _ in items] for review_id, items in claims.items()}


def appendix_panel(panel, claims, n_reviews):
    df = assign(panel, claims)
    groups = review_groups(df.review, df.cluster, n_reviews)
    shared = set().union(*(a & b for a, b in combinations(groups, 2)))
    kept = df[~df.appendix]
    return dict(
        hit=mean_overlap(groups),
        hit_without=mean_overlap(review_groups(kept.review, kept.cluster, n_reviews)),
        appendix_claims=df.appendix.mean(),
        appendix_shared=len(shared & set(df.cluster[df.appendix])) / len(shared) if shared else np.nan,
    )


def claim_share(counts):
    share = counts["sum"].sum() / counts["size"].sum()
    se = np.sqrt(((counts["sum"] - share * counts["size"]) ** 2).sum()) / counts["size"].sum()
    return dict(reviews=len(counts), claims=int(counts["size"].sum()), share=share * 100, se=se * 100)


def appendix_claims(model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR):
    main = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet").query("n_claims > 0"))
    clusters = hub.clusters(model, panel_ids=main.panel_id)
    claims = appendix_marks(hub.panel_claims(model, clusters, ["quote", "claim"]))
    rows, marks = [], {}
    for p in tqdm(main.itertuples(), total=len(main)):
        rows.append(dict(conf=p.conf, year=p.year, **appendix_panel(clusters[p.panel_id], claims, p.n_reviews)))
        for review_id in sorted(r for r in clusters[p.panel_id]["review_ids"] if claims.get(r)):
            marks.setdefault(review_id, [m["appendix"] for m in claims[review_id]])
    panels = pd.DataFrame(rows)
    panels["removed_change"] = panels.hit_without - panels.hit
    cols = ["hit", "hit_without", "removed_change", "appendix_claims", "appendix_shared"]
    by_year = panels.groupby(["conf", "year"])[cols]
    out = pd.concat([by_year.mean(), by_year.sem().add_suffix("_se")], axis=1) * 100
    out.insert(0, "papers", panels.groupby(["conf", "year"]).size())
    out = out.reset_index()
    out.round(3).to_csv(Path(reports) / f"appendix_panels_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    counts = pd.DataFrame(
        {"size": [len(m) for m in marks.values()], "sum": [sum(m) for m in marks.values()]},
        index=pd.Index(list(marks), name="review_id"),
    )
    flags = hub.meld_flags(["conf", "year", "flag"]).join(hub.detector_flags("editlens"))
    counts = counts.join(flags, how="inner")
    shares = []
    for detector, column in (("meld", "flag"), ("editlens", "editlens_flag")):
        for (conf, year, flag), cell in counts.dropna(subset=[column]).groupby(["conf", "year", column]):
            shares.append(dict(detector=detector, conf=conf, year=year, flagged=flag) | claim_share(cell))
    shares = pd.DataFrame(shares)
    shares.round(3).to_csv(Path(reports) / f"appendix_claims_{model}.csv", index=False)
    print(shares.round(2).to_string(index=False))


def coverage(model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR, tex=settings.TABLES_DIR / "coverage.tex"):
    reviews = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"review_groups_{model}.parquet"))
    rows = []
    for (year, _), panel in tqdm(reviews[reviews.conf == "ICLR"].groupby(["year", "panel_id"], sort=False)):
        groups = [set(g) for g in panel.groups]
        if len(groups) < 3:
            continue
        trios = list(combinations(groups, 3))
        rows.append(
            dict(
                year=year,
                problems=np.mean([len(a | b | c) for a, b, c in trios]),
                per_review=np.mean([len(g) for g in groups]),
                new_by_third=np.mean([len(t[i] - t[i - 1] - t[i - 2]) for t in trios for i in range(3)]),
                shared_per_pair=np.mean([len(a & b) for a, b in combinations(groups, 2)]),
            )
        )
    panels = pd.DataFrame(rows)
    shown = ["per_review", "problems", "new_by_third", "shared_per_pair"]
    by_year = panels.groupby("year")[shown]
    out = pd.concat([by_year.mean(), by_year.sem().add_suffix("_se")], axis=1)
    out.insert(0, "papers", panels.groupby("year").size())
    out = out.reset_index()
    out.round(3).to_csv(Path(reports) / f"coverage_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    heads = [r"Per\\review", "Distinct", r"New in a\\third review", r"Shared\\per pair"]
    lines = [
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l" + "c" * len(heads) + "@{}}",
        r"\toprule",
        rf" & \multicolumn{{{len(heads)}}}{{c}}{{Claims}} \\",
        rf"\cmidrule{{2-{len(heads) + 1}}}",
        "Year & " + " & ".join(rf"\shortstack{{{head}}}" for head in heads) + r" \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        mark = r"\rlap{$^\dagger$}" if r.year >= settings.ICLR_SNAPSHOT_YEARS[0] else ""
        lines.append(f"{r.year}{mark} & " + " & ".join(f"{getattr(r, c):.2f}" for c in shown) + r" \\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular*}"]) + "\n")


PASS1_SAMPLES = {"ml": ["main", "icml"], "journal": ["journal"]}
PASS1_VARIANTS = {"ml": ["noq", "nocq"], "journal": []}


ALL_PAIR_TYPES = ("HH", *PAIR_TYPES)
DECOMPOSITION_TERMS = ("change", "background", "llm", "composition", "flagged_excess")
TABLE_TERMS = ("change", "background", "composition", "flagged_excess")


def type_totals(weights, hits):
    share = weights.sum(axis=0) / len(weights)
    return share, 100 * hits.sum(axis=0) / weights.sum(axis=0)


def shift_share(before, after):
    (share_a, mean_a), (share_b, mean_b) = before, after
    mean_a, mean_b = np.where(np.isnan(mean_a), mean_b, mean_a), np.where(np.isnan(mean_b), mean_a, mean_b)
    change = np.nan_to_num(share_b * mean_b).sum() - np.nan_to_num(share_a * mean_a).sum()
    composition = np.nan_to_num((share_b - share_a) * (mean_a + mean_b) / 2).sum()
    background = mean_b[0] - mean_a[0]
    llm = change - background
    return np.array([change, background, llm, composition, llm - composition])


PAIR_TYPE_HEADS = ["None", "One", "Two"]
CLEAN_HEAD = r"None, no\\flag in\\paper"
DECOMPOSITION_HEADS = ["Total", r"Unflagged\\pairs", r"Shift of\\types", r"Flagged,\\above\\unflagged"]


def decomposition_table(out, tex):
    heads = [rf"\shortstack[r]{{{head}}}" for head in [*PAIR_TYPE_HEADS, *PAIR_TYPE_HEADS, CLEAN_HEAD, *DECOMPOSITION_HEADS]]
    lines = [
        r"\begin{tabular}{@{}llrrrrrrrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Share of pairs (\%)} & \multicolumn{4}{c}{Overlap (\%)} & "
        r"\multicolumn{4}{c}{Change from year before (points)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}",
        "Source & Year & " + " & ".join(heads) + r" \\",
        r"\midrule",
    ]
    for conf, group in out.groupby("conf", sort=False):
        for k, r in enumerate(group.itertuples()):
            cells = [conf if k == 0 else "", str(r.year)]
            cells += [f"{getattr(r, f'share_{t}'):.1f}" for t in ALL_PAIR_TYPES]
            cells += [f"{getattr(r, f'overlap_{t}'):.1f}" for t in ALL_PAIR_TYPES]
            cells.append(f"{r.overlap_clean:.1f}")
            cells += ["--" if pd.isna(r.change) else f"{getattr(r, name):+.2f}" for name in TABLE_TERMS]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    Path(tex).write_text("\n".join(lines[:-1] + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def decomposition_table_from_report(
    model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR, tex=settings.TABLES_DIR / "decomposition.tex"
):
    decomposition_table(pd.read_csv(Path(reports) / f"decomposition_{model}.csv"), tex)


def decomposition_rows(pairs, flag):
    spans = {}
    for conf, first, last in settings.GAP_CELLS.values():
        spans[conf] = (min(first, spans.get(conf, (first, last))[0]), max(last, spans.get(conf, (first, last))[1]))
    rng = np.random.default_rng(settings.REFERENCE_BOOTSTRAP_SEED)
    rows = []
    for conf, (first, last) in spans.items():
        previous = None
        for year in range(first, last + 1):
            cell = pairs[(pairs.conf == conf) & (pairs.year == year)]
            types = pair_type(cell, flag)
            weight = 1 / cell.groupby("panel_id").hit.transform("size")
            weights = (
                weight.groupby([cell.panel_id, types]).sum().unstack(fill_value=0).reindex(columns=ALL_PAIR_TYPES, fill_value=0)
            )
            hits = (
                (weight * cell.hit)
                .groupby([cell.panel_id, types])
                .sum()
                .unstack(fill_value=0)
                .reindex(columns=ALL_PAIR_TYPES, fill_value=0)
            )
            weights, hits = weights.to_numpy(), hits.to_numpy()
            share, mean = type_totals(weights, hits)
            draws = [rng.integers(len(weights), size=len(weights)) for _ in range(settings.REFERENCE_BOOTSTRAPS)]
            resampled = [type_totals(weights[d], hits[d]) for d in draws]
            row = dict(conf=conf, year=year, papers=len(weights))
            row.update({f"share_{t}": 100 * v for t, v in zip(ALL_PAIR_TYPES, share, strict=True)})
            row.update({f"overlap_{t}": v for t, v in zip(ALL_PAIR_TYPES, mean, strict=True)})
            clean = weights[:, 1:].sum(axis=1) == 0
            row["overlap_clean"] = 100 * hits[clean, 0].sum() / weights[clean, 0].sum()
            if previous is not None:
                terms = shift_share(previous[0], (share, mean))
                errors = np.std([shift_share(a, b) for a, b in zip(previous[1], resampled, strict=True)], axis=0)
                row.update(dict(zip(DECOMPOSITION_TERMS, terms, strict=True)))
                row.update({f"{name}_se": e for name, e in zip(DECOMPOSITION_TERMS, errors, strict=True)})
            previous = ((share, mean), resampled)
            rows.append(row)
    return pd.DataFrame(rows)


def decomposition(
    model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR, tex=settings.TABLES_DIR / "decomposition.tex"
):
    meld = hub.meld_flags()
    pairs = analysis_pairs(model)
    out = decomposition_rows(pairs, meld.flag.astype(int))
    out.to_csv(Path(reports) / f"decomposition_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    decomposition_table(out, tex)


Z95 = 1.96


def second_detector_table(out, tex):
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{MELD} & \multicolumn{2}{c}{GigaCheck} \\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r"Reviews & Papers & Gap (points) & Pairs & Gap (points) & Pairs \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        cells = [r.cell.replace("–", "--"), f"{r.papers:,}"]
        for name in ["meld", "gigacheck"]:
            est, se = getattr(r, name), getattr(r, f"{name}_se")
            cells += [f"{est:+.1f} [{est - Z95 * se:+.1f}, {est + Z95 * se:+.1f}]", f"{getattr(r, f'{name}_pairs'):,}"]
        lines.append(" & ".join(cells) + r" \\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def second_detector(
    model=settings.DEFAULT_MODEL,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "second_detector.tex",
    first_year=2024,
):
    flags = {"meld": hub.meld_flags().flag.astype(int), "gigacheck": hub.detector_flags("gigacheck").astype(int)}
    pairs = analysis_pairs(model)
    journals = pairs[~pairs.conf.isin(settings.ML_VENUES) & (pairs.year >= first_year)]
    cells = gap_cells(pairs) | {f"Journals {first_year}–{journals.year.max() % 100}": journals}
    rows = []
    for label, cell in cells.items():
        row = dict(cell=label, papers=cell.panel_id.nunique())
        for name, flag in flags.items():
            gap = direct_gap(cell, flag)
            row |= {name: gap["two"], f"{name}_se": gap["two_se"], f"{name}_pairs": int((pair_counts(cell, flag) == 2).sum())}
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / f"second_detector_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    second_detector_table(out, tex)


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
        hit_orig, _ = matched_claims(panel["clusters"], n_orig, n_rw)
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
    out=settings.REWRITE_TEST_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "rewrite_test.tex",
):
    out = Path(out)
    claims = {rid: row["claims"] for rid, row in load_jsonl(out / f"claims_{model}.jsonl", "review_id").items()}
    clusters = load_jsonl(out / f"clusters_{model}.jsonl", "panel_id")
    papers = list(read_jsonl(out / "papers.jsonl"))
    main = hub.clusters(model, panel_ids={p["panel_id"] for p in papers})
    rows, all_pairs = [], []
    for condition, c in settings.REWRITE_CONDITIONS.items():
        pairs = rewrite_pairs(out, claims, clusters, main, c.panel).assign(condition=condition)
        all_pairs.append(pairs)
        luna_kept = preservation(clusters, claims, c.preserve)
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
        ("link_ratio", tp + fp, tp + fn),
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
    names = {"same": "Same", "cross": "Cross"}
    titles = [title.partition("|") for _, title, _, _ in columns]
    header = [" & & & & " + " & ".join(top for top, _, _ in titles) + r" \\"] if any(sep for _, sep, _ in titles) else []
    header.append("Panels & Source & Detector stratum & $n$ & " + " & ".join(bottom or top for top, _, bottom in titles) + r" \\")
    lines = [r"\begin{tabular}{@{}lll" + "r" * (len(columns) + 1) + "@{}}", r"\toprule", *header, r"\midrule"]
    for r in out[(out.split == "all") & out.conf.isin(["all", *settings.ML_VENUES])].itertuples():
        cell = (r.years if r.stratum == "all" else r.stratum).replace("all", "All")
        values = []
        for name, _, digits, with_se in columns:
            text = f"{getattr(r, name):.{digits}f}"
            values.append(text + f" ({getattr(r, name + '_se'):.{digits}f})" if with_se else text)
        lines.append(f"{names[r.kind]} & {r.conf.replace('all', 'All')} & {cell} & {r.panels} & " + " & ".join(values) + r" \\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


PANEL_KEYS = ("panel_id", "kind", "conf", "year", "stratum", "split")


def panel_meta(panel):
    return {key: panel[key] for key in PANEL_KEYS}


def gold_panels(gold):
    panels = read_jsonl(Path(gold) / "panels_tune.jsonl")
    return [p | {"stratum": "flagged" if p["stratum"] == "written" else p["stratum"]} for p in panels]


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


EXTRACTION_COLUMNS = [
    ("claims_model", "Claims|Luna", 1, False),
    ("claims_council", "Claims|council", 1, False),
    ("f1_model", "Span F1|Luna", 1, False),
    ("f1_council", "Span F1|council", 1, False),
    ("cover_recall_model", "Member in|Luna's list", 1, False),
    ("cover_precision_model", "Luna in|member's list", 1, False),
    ("cover_council", "Member in|member's list", 1, False),
]


def extraction_table(out, tex):
    whole = (out.years == "all") & ((out.kind == "same") | ((out.conf == "all") & (out.stratum == "all")))
    write_cell_table(out[whole], tex, EXTRACTION_COLUMNS)


def write_extraction(reviews, model, reports, tex):
    keys = list(PANEL_KEYS)
    metrics = [c for c in reviews.columns if c not in keys + ["review_id"]]
    by_panel = reviews.groupby(keys, sort=False)
    panels = by_panel[metrics].mean().reset_index().assign(reviews=by_panel.size().to_numpy())
    summary = []
    for kind, by_kind in panels.groupby("kind", sort=False):
        for label, cell in report_cells(by_kind):
            row = dict(kind=kind, **label, panels=len(cell), reviews=int(cell.reviews.sum()))
            for name in metrics:
                row.update(mean_interval(cell[name], name))
            summary.append(row)
    out = pd.DataFrame(summary)
    out.round(3).to_csv(Path(reports) / f"extraction_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    extraction_table(out, tex)


def extraction_table_from_report(
    model=settings.DEFAULT_MODEL,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "extraction.tex",
):
    meta = pd.DataFrame([panel_meta(p) for p in gold_panels(gold)])
    reviews = pd.read_csv(Path(reports) / f"extraction_reviews_{model}.csv")
    write_extraction(meta.merge(reviews.drop(columns=list(PANEL_KEYS[1:])), on="panel_id"), model, reports, tex)


def extraction(
    model=settings.DEFAULT_MODEL,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "extraction.tex",
    council=settings.GOLD_COUNCIL,
):
    gold = Path(gold)
    panels = gold_panels(gold)
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
    write_extraction(reviews, model, reports, tex)


def panel_overlap(panel, claims, clusters):
    row = clusters.get(panel["panel_id"], {})
    if "clusters" not in row or any(rid not in claims for rid in panel["review_ids"]):
        return np.nan
    return 100 * strict_overlap(panel["review_ids"], claims, row["clusters"])


def pipelines(
    model=settings.DEFAULT_MODEL,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    contrasts_tex=settings.TABLES_DIR / "pipeline_contrasts.tex",
    council=settings.GOLD_COUNCIL,
):
    gold = Path(gold)
    panels = gold_panels(gold)
    runs = {
        model: (extract.load_claims(gold / f"claims_{model}.jsonl"), load_jsonl(gold / f"clusters_{model}.jsonl", "panel_id"))
    }
    for member in council:
        runs[member] = (
            extract.load_claims(gold / f"claims_{member}.jsonl"),
            load_jsonl(gold / f"clusters_{member}_own.jsonl", "panel_id"),
        )
    pair_gaps = own_pair_gaps(panels, runs, hub.meld_flags().flag.astype(int))
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
            summary.append(row)
    out = pd.DataFrame(summary)
    out.round(3).to_csv(Path(reports) / f"pipelines_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    contrasts = pd.DataFrame([pipeline_contrasts(panels, name) | pair_gaps[name] for name in runs])
    contrasts.round(3).to_csv(Path(reports) / f"pipelines_contrasts_{model}.csv", index=False)
    print(contrasts.round(2).to_string(index=False))
    write_contrast_table(contrasts, contrasts_tex)


def own_pair_gaps(panels, runs, flag):
    rows = []
    for panel in panels:
        pid, ids = panel["panel_id"], panel["review_ids"]
        if panel["kind"] != "same" or panel["year"] < settings.FLAG_FROM:
            continue
        if not all(
            "clusters" in clusters.get(pid, {}) and all(rid in claims for rid in ids) for claims, clusters in runs.values()
        ):
            continue
        sets = {name: grouped(ids, claims, clusters[pid]["clusters"]) for name, (claims, clusters) in runs.items()}
        for i, j in combinations(range(len(ids)), 2):
            rows.append(dict(panel_id=pid, a=ids[i], b=ids[j], **{name: pair_overlap(s[i], s[j]) for name, s in sets.items()}))
    pairs = pd.DataFrame(rows)
    gaps = {name: fe_gap(pairs.assign(hit=pairs[name]), flag) for name in runs}
    return {
        name: dict(pair_gap=gap["two"], pair_gap_se=gap["two_se"], pair_gap_panels=pairs.panel_id.nunique())
        for name, gap in gaps.items()
    }


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
    first, last = settings.ICLR_SNAPSHOT_YEARS

    def gap(frame, a, b):
        return frame[frame.stratum == a][column].mean() - frame[frame.stratum == b][column].mean()

    def slope(frame):
        return np.polyfit(frame.year.to_numpy(float), frame[column].to_numpy(float), 1)[0]

    def step(frame):
        return frame[frame.year == last][column].mean() - frame[frame.year == first][column].mean()

    row = dict(pipeline=name, panels=len(same), level=same[column].mean(), cross=panels[panels.kind == "cross"][column].mean())
    row.update(bootstrap_interval(same, lambda f: gap(f, "flagged", "plain"), "flag_gap"))
    row.update(bootstrap_interval(iclr, slope, "iclr_slope"))
    row.update(bootstrap_interval(iclr, step, "iclr_step"))
    return row


def write_contrast_table(contrasts, tex):
    first, last = settings.ICLR_SNAPSHOT_YEARS
    lines = [
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Overlap (\%)} & \multicolumn{2}{c}{Contrast (points)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Pipeline & \shortstack[r]{Same\\paper} & \shortstack[r]{Cross\\paper} & \shortstack[r]{Flagged\\gap}"
        rf" & \shortstack[r]{{ICLR {last}\\$-$ {first}}} \\",
        r"\midrule",
    ]
    for r in contrasts.itertuples():
        lines.append(
            f"{settings.MODEL_LABELS[r.pipeline]} & {r.level:.1f} & {r.cross:.1f} & {r.pair_gap:+.1f} & {r.iclr_step:+.1f} \\\\"
        )
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular*}"]) + "\n")


def unique_claim_group_ids(clusters):
    counts = Counter(cid for cluster in clusters for cid in cluster["claim_ids"])
    return {cid: group for cid, group in claim_group_ids(clusters).items() if counts[cid] == 1}


def council_pair_label(groupings, a, b):
    if len(groupings) != len(settings.GOLD_COUNCIL) or any(a not in g or b not in g for g in groupings):
        return "invalid", None
    votes = [g[a] == g[b] for g in groupings]
    return ("unanimous" if len(set(votes)) == 1 else "split"), 2 * sum(votes) > len(votes)


def panel_counts(voted):
    truth, predicted = voted.consensus.astype(bool), voted.predicted.astype(bool)
    hits = pd.DataFrame(
        {"pair_tp": truth & predicted, "pair_fp": ~truth & predicted, "pair_fn": truth & ~predicted, "pair_candidates": 1},
        index=voted.index,
    )
    return hits.groupby(voted.panel_id.to_numpy(), sort=False).sum()[PAIR_COUNTS]


def council_agreement(
    model=settings.DEFAULT_MODEL,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "council.tex",
    council=settings.GOLD_COUNCIL,
):
    gold = Path(gold)
    panels = gold_panels(gold)
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
                    a=ids[ia],
                    b=ids[ib],
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
            voted = cell[cell.status != "invalid"]
            summary.append(
                dict(
                    kind=kind,
                    **label,
                    panels=cell.panel_id.nunique(),
                    reviews=sum(n_reviews[pid] for pid in set(cell.panel_id)),
                    candidate_pairs=len(cell),
                    invalid_pairs=int((cell.status == "invalid").sum()),
                    split_pairs=int((cell.status == "split").sum()),
                    voted_pairs=len(voted),
                    model_positives=int(cell.predicted.sum()),
                    retained_model_positives=int(voted.predicted.sum()),
                    **pair_detection_summary(panel_counts(voted)),
                )
            )
    out = pd.DataFrame(summary)
    out.round(3).to_csv(Path(reports) / f"council_agreement_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    columns = [
        ("candidate_pairs", "Pairs", 0, False),
        ("voted_pairs", "With three votes", 0, False),
        ("pair_precision", "Precision", 1, False),
        ("pair_recall", "Recall", 1, False),
        ("pair_f1", "F1", 1, False),
        ("pair_link_ratio", "Links per 100", 0, False),
    ]
    write_cell_table(out[(out.kind == "same") & (out.years == "all")], tex, columns)


def council_pair_types(model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR):
    pairs = pd.read_csv(Path(reports) / f"council_agreement_pairs_{model}.csv", low_memory=False)
    meld = hub.meld_flags()
    flagged = pairs[(pairs.kind == "same") & (pairs.stratum == "flagged") & (pairs.status != "invalid")]
    types = pair_type(flagged, meld.flag.astype(int)).to_numpy()
    by_type = pd.DataFrame(
        [dict(pair_type=t, **pair_detection_summary(panel_counts(cell))) for t, cell in flagged.groupby(types)]
    )
    by_type.round(3).to_csv(Path(reports) / f"council_pair_types_{model}.csv", index=False)
    print(by_type.round(2).to_string(index=False))


MAJORITY = "majority"
SWAP_LABELS = {**settings.MODEL_LABELS, MAJORITY: "Council majority"}


def majority_clusters(cids, votes):
    links = [(i, j) for (i, a), (j, b) in combinations(enumerate(cids), 2) if council_pair_label(votes, a, b)[1]]
    rows, cols = zip(*links) if links else ((), ())
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cids), len(cids)))
    n, component = connected_components(graph, directed=False)
    return [{"claim_ids": [cid for cid, k in zip(cids, component) if k == c]} for c in range(n)]


def grouper_pairs(model, gold, council):
    panels = [p for p in gold_panels(gold) if p["kind"] == "same" and p["year"] >= settings.FLAG_FROM]
    claims = extract.load_claims(gold / f"claims_{model}.jsonl")
    ours = load_jsonl(gold / f"clusters_{model}.jsonl", "panel_id")
    raw = {(row["panel_id"], row["model"]): row for row in read_jsonl(gold / "clusters_council.jsonl")}
    rows = []
    for panel in panels:
        pid, ids = panel["panel_id"], panel["review_ids"]
        proposals = {model: ours.get(pid, {}), **{member: raw.get((pid, member), {}) for member in council}}
        if any("clusters" not in proposal for proposal in proposals.values()):
            continue
        clusters = {name: proposal["clusters"] for name, proposal in proposals.items()}
        cids = [f"R{j}-{k}" for j, rid in enumerate(ids) for k in range(len(claims[rid]))]
        clusters[MAJORITY] = majority_clusters(cids, [unique_claim_group_ids(clusters[member]) for member in council])
        sets = {name: grouped(ids, claims, groups) for name, groups in clusters.items()}
        for i, j in combinations(range(len(ids)), 2):
            rows.append(
                dict(
                    panel_id=pid,
                    stratum=panel["stratum"],
                    a=ids[i],
                    b=ids[j],
                    **{name: pair_overlap(s[i], s[j]) for name, s in sets.items()},
                )
            )
    return pd.DataFrame(rows)


def swap_row(pairs, flag, panels, model, name):
    def panel_gap(frame):
        return 100 * (frame[frame.stratum == "flagged"][name].mean() - frame[frame.stratum == "plain"][name].mean())

    row = dict(grouper=name, panels=len(panels))
    row.update(bootstrap_interval(panels, panel_gap, "panel_gap"))
    outcomes = {"": pairs[name]}
    if name != model:
        outcomes["diff_"] = pairs[model] - pairs[name]
    for prefix, outcome in outcomes.items():
        gaps = fe_gap(pairs.assign(hit=outcome), flag)
        row.update({f"{prefix}{k}": v for k, v in gaps.items()})
    return row


def grouper_swap_table(out, tex):
    lines = [
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Within-paper gap (points)} \\",
        r"\cmidrule(l){2-3}",
        r"Grouper & One flagged & Two flagged \\",
        r"\midrule",
    ]
    for r in out.itertuples():
        cells = []
        for name in ["one", "two"]:
            value, se = getattr(r, name), getattr(r, name + "_se")
            cells.append(f"{value:+.1f} [{value - Z95 * se:.1f}, {value + Z95 * se:.1f}]")
        lines.append(f"{SWAP_LABELS[r.grouper]} & " + " & ".join(cells) + r" \\")
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular*}"]) + "\n")


def grouper_swap(
    model=settings.DEFAULT_MODEL,
    gold=settings.GOLD_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "grouper_swap.tex",
    council=settings.GOLD_COUNCIL,
):
    pairs = grouper_pairs(model, Path(gold), council)
    flag = hub.meld_flags().flag.astype(int)
    groupers = [model, *council, MAJORITY]
    panels = pairs.groupby(["panel_id", "stratum"])[groupers].mean().reset_index()
    out = pd.DataFrame([swap_row(pairs, flag, panels, model, name) for name in groupers])
    out.to_csv(Path(reports) / f"grouper_swap_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    grouper_swap_table(out, tex)


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
    rows = [dict(review_id=rid, **quote_row(text, claims)) for rid, text, claims in hub.texts_with_claims(model, ["quote"])]
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
    gold = Path(gold)
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
            "forms_table": forms_table_from_report,
            "detector_placebo": detector_placebo,
            "rq1_evidence": rq1_evidence,
            "journal_gaps": journal_gaps,
            "extraction": extraction,
            "extraction_table": extraction_table_from_report,
            "council_agreement": council_agreement,
            "council_pair_types": council_pair_types,
            "grouper_swap": grouper_swap,
            "pipelines": pipelines,
            "identification": identification,
            "abstract_papers": abstract_papers,
            "appendix_claims": appendix_claims,
            "pair_types": pair_types,
            "pair_type_sizes": pair_type_sizes,
            "pair_type_sizes_table": pair_type_sizes_table_from_report,
            "journal_subsets": journal_subsets,
            "coverage": coverage,
            "decomposition": decomposition,
            "decomposition_table": decomposition_table_from_report,
            "second_detector": second_detector,
            "rewrite_test": rewrite_test,
            "repro": repro,
            "quotes": quotes,
            "external": external,
        }
    )
