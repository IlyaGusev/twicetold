from itertools import combinations
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

import hub
from outcomes import in_window, scored_pairs, scored_review_ids
from settings import settings
from validate import pair_counts

Z95 = 1.96
MAX_CELL_SIZE = 25
MIN_NOVELTY_PAPERS = 30
JOURNALS = "journals"
PRE_PERIOD = tuple(range(2019, 2024))
TRENDS = [
    (
        "ICLR, {} to {}".format(*settings.ICLR_SNAPSHOT_YEARS),
        "ICLR",
        settings.ICLR_SNAPSHOT_YEARS[:1],
        settings.ICLR_SNAPSHOT_YEARS[1],
    ),
    ("ICLR, 2025 to 2026", "ICLR", (2025,), 2026),
    ("NeurIPS, 2024 to 2025", "NeurIPS", (2024,), 2025),
    ("ICML, 2025 to 2026", "ICML", (2025,), 2026),
    ("F1000, 2019--23 to 2026", "F1000", PRE_PERIOD, 2026),
    ("PLOS, 2019--23 to 2026", "PLOS", PRE_PERIOD, 2026),
]
GAPS = [
    ("ICLR, two flagged", "ICLR", settings.ICLR_GAP_YEARS, "flag"),
    ("NeurIPS, two flagged", "NeurIPS", (2024, 2025), "flag"),
    ("ICML, two flagged", "ICML", (2025, 2026), "flag"),
    ("Journals, two flagged", JOURNALS, (2024, 2026), "flag"),
]
MEASURES = {
    "hit": ("Overlap coef.", "points", 1),
    "jaccard": ("Jaccard", "points", 1),
    "dice": ("Dice", "points", 1),
    "shared": ("Shared", "per pair", 2),
    "shared_linear": ("Shared, linear sizes", "per pair", 2),
    "shared_cells": ("Shared, same sizes", "per pair", 2),
    "novelty": ("New problems", "per review", 2),
}


def unique_counts(groups):
    return [len(g - set().union(*(h for j, h in enumerate(groups) if j != i))) for i, g in enumerate(groups)]


def third_review_novelty(groups):
    return np.mean([n for trio in combinations(groups, 3) for n in unique_counts(list(trio))])


def pair_measures(pairs):
    nonempty = pairs.union.where(pairs.union > 0)
    return pairs.assign(
        hit=pairs.hit * 100,
        jaccard=(pairs.shared / nonempty).fillna(0) * 100,
        dice=(2 * pairs.shared / (pairs.smaller + pairs.larger).where(pairs.union > 0)).fillna(0) * 100,
    )


def cell_key(*sizes):
    key = pd.Series("", index=sizes[0].index)
    for size in sizes:
        key = key + "_" + size.clip(upper=MAX_CELL_SIZE).astype(str)
    return key


def size_cells(*sizes):
    return pd.get_dummies(cell_key(*sizes), prefix="cell", drop_first=True, dtype=float)


def pair_specs(frame):
    cells = size_cells(frame.smaller, frame.larger)
    specs = {name: (name, []) for name in ("hit", "jaccard", "dice", "shared")}
    specs["shared_linear"] = ("shared", ["smaller", "larger"])
    specs["shared_cells"] = ("shared", list(cells.columns))
    return frame.join(cells), specs


def trend_specs(frame):
    specs = {name: (name, [], None) for name in ("hit", "jaccard", "dice", "shared")}
    specs["shared_linear"] = ("shared", ["smaller", "larger"], None)
    specs["shared_cells"] = ("shared", [], "cell")
    return frame.assign(cell=cell_key(frame.smaller, frame.larger)), specs


def in_venue(frame, venue):
    return ~frame.conf.isin(settings.ML_VENUES) if venue == JOURNALS else frame.conf == venue


def cluster_fit(model):
    return lambda groups: model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def trend(frame, base_years, last_year, outcome, controls=(), absorbed=None):
    d = frame[frame.year.isin([*base_years, last_year])]
    controls = [c for c in controls if d[c].nunique() > 1]
    weight = 1 / (d.groupby("panel_id").panel_id.transform("size") * d.groupby("year").panel_id.transform("nunique"))
    late = (d.year == last_year) * 1.0
    if absorbed is None:
        X = sm.add_constant(d[list(controls)].assign(late=late))
        fit = cluster_fit(sm.WLS(d[outcome], X, weights=weight))(d.panel_id)
        return fit.params["late"], fit.bse["late"], d.panel_id.nunique()
    cells = d[absorbed]
    cell_weight = weight.groupby(cells).transform("sum")

    def centred(values):
        return values - (values * weight).groupby(cells).transform("sum") / cell_weight

    fit = cluster_fit(sm.WLS(centred(d[outcome]), centred(late).rename("late"), weights=weight))(d.panel_id)
    lost = np.sqrt((len(d) - 1) / (len(d) - 1 - cells.nunique()))
    return fit.params["late"], fit.bse["late"] * lost, d.panel_id.nunique()


def within_paper(frame, outcome, terms, controls=()):
    X = frame[[*terms, *controls, outcome]]
    X = X - X.groupby(frame.panel_id.to_numpy()).transform("mean")
    controls = [c for c in controls if X[c].abs().max() > 1e-12]
    return cluster_fit(sm.OLS(X[outcome], X[[*terms, *controls]]))(frame.panel_id)


def pair_gap(pairs, flag, outcome, controls):
    n_labeled = pair_counts(pairs, flag).to_numpy()
    d = pairs.assign(one=(n_labeled == 1) * 1.0, two=(n_labeled == 2) * 1.0)
    fit = within_paper(d, outcome, ["one", "two"], controls)
    papers = pd.Series(n_labeled, index=d.index).groupby(d.panel_id).nunique().gt(1).sum()
    return fit.params["two"], fit.bse["two"], papers


def review_gap(reviews, flag):
    d = reviews.assign(labeled=reviews.review_id.map(flag).fillna(0).astype(float))
    cells = size_cells(d.n_groups)
    fit = within_paper(d.join(cells), "unique", ["labeled"], list(cells.columns))
    return fit.params["labeled"], fit.bse["labeled"], d.groupby("panel_id").labeled.nunique().gt(1).sum()


def review_table(model):
    rows = []
    reviews = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"review_groups_{model}.parquet"))
    for panel_id, panel in tqdm(reviews.groupby("panel_id", sort=False), desc="novelty"):
        groups = [set(g) for g in panel.groups]
        third = third_review_novelty(groups) if len(groups) >= 3 else np.nan
        for rid, g, unique in zip(panel.review_id, groups, unique_counts(groups)):
            rows.append(dict(panel_id=panel_id, review_id=rid, n_groups=len(g), unique=unique, third=third))
    out = pd.DataFrame(rows)
    meta = reviews[["review_id", "panel_id", "conf", "year"]]
    return out.merge(meta, on=["panel_id", "review_id"])


def summarize(estimate, se, papers, digits, sign):
    return dict(est=estimate, se=se, papers=int(papers), agrees=bool(sign * (estimate - sign * Z95 * se) > 0), digits=digits)


def estimates(model):
    meld = hub.meld_flags()
    labels = {"flag": meld.flag.astype(int), "written": meld.written.astype(int)}
    pairs = pair_measures(in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet")))
    reviews = review_table(model)
    papers = reviews.drop_duplicates("panel_id").dropna(subset=["third"])
    rows = []
    for claim, venue, base_years, last_year in tqdm(TRENDS, desc="trends"):
        frame, specs = trend_specs(pairs[in_venue(pairs, venue)])
        for name, (outcome, controls, absorbed) in specs.items():
            fit = trend(frame, base_years, last_year, outcome, controls, absorbed)
            rows.append(dict(kind="trend", claim=claim, measure=name, **summarize(*fit, MEASURES[name][2], 1)))
        venue_papers = papers[in_venue(papers, venue)]
        counts = venue_papers.year.value_counts()
        if counts.reindex([*base_years, last_year]).fillna(0).min() >= MIN_NOVELTY_PAPERS:
            fit = trend(venue_papers, base_years, last_year, "third")
            rows.append(dict(kind="trend", claim=claim, measure="novelty", **summarize(*fit, 2, -1)))
    for claim, venue, years, label in tqdm(GAPS, desc="gaps"):
        frame, specs = pair_specs(scored_pairs(pairs[in_venue(pairs, venue) & pairs.year.between(*years)]))
        for name, (outcome, controls) in specs.items():
            fit = pair_gap(frame, labels[label], outcome, controls)
            rows.append(dict(kind="gap", claim=claim, measure=name, **summarize(*fit, MEASURES[name][2], 1)))
        venue_reviews = reviews[in_venue(reviews, venue) & reviews.year.between(*years)]
        venue_reviews = venue_reviews[venue_reviews.review_id.astype(object).isin(scored_review_ids())]
        fit = review_gap(venue_reviews, labels[label])
        rows.append(dict(kind="gap", claim=claim, measure="novelty", **summarize(*fit, 2, -1)))
    return pd.DataFrame(rows)


def cell(row):
    if row is None:
        return "--"
    text = f"{row.est:+.{row.digits}f}"
    return text if row.agrees else text + r"$^\dagger$"


def latex(out):
    names = list(MEASURES)
    header = " & ".join(rf"\shortstack{{{MEASURES[m][0].replace(', ', r',\\')}}}" for m in names)
    units = " & ".join(MEASURES[m][1] for m in names)
    lines = [
        r"\begin{tabular}{@{}l" + "r" * len(names) + "@{}}",
        r"\toprule",
        rf"Claim & {header} \\",
        rf" & {units} \\",
    ]
    for kind, title in [("trend", "Change over time"), ("gap", "Within-paper gap against unlabeled pairs or reviews")]:
        lines += [r"\midrule", rf"\multicolumn{{{len(names) + 1}}}{{@{{}}l}}{{\textbf{{{title}}}}} \\"]
        for claim, rows in out[out.kind == kind].groupby("claim", sort=False):
            by_measure = {r.measure: r for r in rows.itertuples()}
            lines.append(claim + " & " + " & ".join(cell(by_measure.get(m)) for m in names) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def measures(model=settings.DEFAULT_MODEL, reports=settings.ANALYSIS_REPORTS_DIR, tex=settings.TABLES_DIR / "measures.tex"):
    out = estimates(model)
    out.to_csv(Path(reports) / f"measures_{model}.csv", index=False)
    print(out.drop(columns="digits").round(3).to_string(index=False))
    Path(tex).write_text(latex(out))


if __name__ == "__main__":
    fire.Fire({"measures": measures})
