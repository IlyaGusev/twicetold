import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

from outcomes import ICLR_FROM, excluded_review_ids, in_window, main_sample_review_ids
from settings import settings

COLUMN_WIDTH = 3.03
PAGE_WIDTH = 6.3
Z95 = 1.96
CHATGPT_BAND = (2022.6, 2026.5)
SAME_VENUE_PLACEBO = "shuffled4"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "errorbar.capsize": 0,
        "pdf.fonttype": 42,
    }
)
COLORS = {
    "ICLR": "#0072B2",
    "NeurIPS": "#D55E00",
    "ICML": "#009E73",
    "F1000": "#CC79A7",
    "PLOS": "#56B4E9",
    "flag": "#E69F00",
    "flag_light": "#F3CF7F",
    "grey": "#8C8C8C",
    "light": "#C8C8C8",
    "ink": "#222222",
    "band": "#EFEFEF",
}
ML_VENUES = ["ICLR", "NeurIPS", "ICML"]
JOURNALS = {"SciPost": "SciPost", "F1000": "F1000", "PLOS": "PLOS", "BMC-Med": "BMC Medicine", "Copernicus": "Copernicus"}
ORDER = ML_VENUES + list(JOURNALS)
JOURNAL_PREFIX = {"F1000": "FResearch", "PLOS": "Plos", "SciPost": "Scipost", "BMC-Med": "Bmc", "Copernicus": "Copernicus"}
SHORT = {
    "+ length and claim count": "+ length, claim count",
    "99th-percentile threshold, ICLR": "99th-pct. threshold",
    "99.5th-percentile threshold, ICLR": "99.5th-pct. threshold",
    "Top-45% scores, ICLR 2020": "ICLR 2020",
    "Top-45% scores, ICLR 2021-2022": "ICLR 2021–22",
    "Top-45% scores, ICLR 2023": "ICLR 2023",
}
CATEGORY_NAMES = {
    "clarity/presentation": "Clarity, presentation",
    "experiments-scope": "Scope of experiments",
    "baselines/comparisons": "Baselines",
    "theory/correctness": "Theory, correctness",
    "reproducibility/details": "Missing details",
    "typo/formatting": "Typos, formatting",
    "novelty": "Novelty",
    "related-work": "Related work",
    "assumptions": "Assumptions",
    "unsupported-claim": "Unsupported claim",
    "ablation": "Ablations",
    "weak-results": "Weak results",
    "compute-cost/scalability": "Compute cost",
    "generalization": "Generalization",
    "statistical-rigor": "Statistical rigor",
    "limitations/societal": "Limitations, societal",
}


CONFERENCE_STYLES = {
    "ICLR": ("ICLR", COLORS["ICLR"], "o", 0),
    "NeurIPS": ("NeurIPS", COLORS["NeurIPS"], "s", 0),
    "ICML": ("ICML", COLORS["ICML"], "D", 0.12),
}
JOURNAL_STYLES = {
    "F1000": ("F1000", COLORS["F1000"], "o", 0),
    "PLOS": ("PLOS", COLORS["PLOS"], "s", 0),
    "Copernicus": ("Copernicus", COLORS["grey"], "^", 0),
    "SciPost": ("SciPost", COLORS["grey"], "v", 0),
    "BMC-Med": ("BMC Medicine", COLORS["grey"], "D", 0),
}


def shade_chatgpt_period(ax):
    ax.axvspan(*CHATGPT_BAND, color=COLORS["band"], zorder=0, lw=0)


def year_ticks(ax, ticks=(2020, 2023, 2026)):
    ax.set_xticks(list(ticks), [f"'{t % 100:02d}" for t in ticks])


def mean_line(ax, df, col, color, ls="-", **kw):
    per_year = df.groupby("year")[col]
    mean = per_year.mean() * 100
    ci = Z95 * per_year.std(ddof=1) / np.sqrt(per_year.count()) * 100
    ax.errorbar(mean.index, mean, yerr=ci, fmt="o", ms=3, color=color, elinewidth=1.0, ls=ls, **kw)
    return pd.DataFrame({"mean": mean, "bar_top": mean + ci})


def overlap_line(ax, by_year, conf, color, dx=0.0):
    rows = by_year[by_year.conf == conf]
    ax.errorbar(rows.year + dx, rows["match"], yerr=Z95 * rows.match_se, fmt="o-", ms=3, color=color, elinewidth=1.0)
    return rows.set_index("year")["match"]


def end_label(ax, x, y, text, color=COLORS["ink"], dx=3, dy=0, ha="left", va="center"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=color, fontsize=7)


def few_ticks(ax, axis="y", n=4):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_locator(MaxNLocator(n, integer=True))


def save(fig, figs, name, width):
    for ax in fig.axes:
        data = ax.dataLim
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        y_outside = data.y0 < y0 or data.y1 > y1
        x_outside = not ax.patches and (data.x0 < x0 or data.x1 > x1)
        if y_outside or x_outside:
            print(f"WARNING {name}: data x {data.x0:.2f}..{data.x1:.2f}, y {data.y0:.2f}..{data.y1:.2f} outside the axis limits")
    figs = Path(figs)
    figs.mkdir(parents=True, exist_ok=True)
    content = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.02)
    if content.width > width + 1e-3:
        print(f"WARNING {name}: content {content.width:.2f} in wider than {width:.2f} in; shrink figsize")
    margin = max(0.0, width - content.width) / 2
    fig.savefig(figs / name, bbox_inches=Bbox([[content.x0 - margin, content.y0], [content.x1 + margin, content.y1]]))
    plt.close(fig)


def meld_flags():
    flags = pd.read_parquet(settings.DATA_DIR / "meld_flags.parquet", columns=["review_id", "conf", "year", "flag", "written"])
    return flags[~flags.review_id.isin(excluded_review_ids(settings.DATA_DIR))]


def rates(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    flags = in_window(meld_flags())
    by_source = flags.groupby(["conf", "year"]).flag.agg(["mean", "size"]).reset_index()
    by_source["match"] = by_source["mean"] * 100
    by_source["match_se"] = np.sqrt(by_source["mean"] * (1 - by_source["mean"]) / by_source["size"]) * 100
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(COLUMN_WIDTH - 0.1, 3.1))
    fig.subplots_adjust(left=0.13, right=0.76, top=0.93, bottom=0.11, hspace=0.42)
    for ax, styles, title in [(top, CONFERENCE_STYLES, "a  ML conferences"), (bottom, JOURNAL_STYLES, "b  Journals")]:
        shade_chatgpt_period(ax)
        overlap_panel(ax, by_source, styles, 2026.65, 5)
        ax.set_ylim(-2, 70)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_ylabel("AI-flagged (%)")
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", labelsize=6)
    bottom.set_xlim(2018.6, CHATGPT_BAND[1])
    bottom.set_xticks(range(2019, 2027))
    top.tick_params(labelbottom=True)
    bottom.set_xlabel("Review year")
    save(fig, figs, "figA_rates.pdf", COLUMN_WIDTH)


PAIR_TYPE_STYLES = {
    "HH": ("Neither flagged", COLORS["grey"], COLORS["grey"], "o"),
    "HL": ("One flagged", COLORS["flag_light"], COLORS["flag"], "s"),
    "LL": ("Both flagged", COLORS["flag"], COLORS["flag"], "D"),
    "WW": ("Both AI-written", COLORS["ink"], COLORS["ink"], "^"),
}


def pair_type_means(model):
    flags = meld_flags().set_index("review_id")
    pairs = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    pairs = pairs[pairs.conf.isin(ML_VENUES) & (pairs.year >= 2024)].copy()
    n_flagged = pairs.a.map(flags.flag).astype(int) + pairs.b.map(flags.flag).astype(int)
    n_written = pairs.a.map(flags.written).astype(int) + pairs.b.map(flags.written).astype(int)
    pairs["pair_type"] = n_flagged.map({0: "HH", 1: "HL", 2: "LL"})
    pairs.loc[n_written == 2, "pair_type"] = "WW"
    per_paper = pairs.groupby(["conf", "pair_type", "panel_id"]).hit.mean() * 100
    means = per_paper.groupby(["conf", "pair_type"]).agg(["mean", "std", "count"]).reset_index()
    years = pairs.groupby("conf").year.agg(["min", "max"])
    means["venue"] = means.conf.map(lambda c: f"{c}\n{years.loc[c, 'min']}–{years.loc[c, 'max'] % 100:02d}")
    means["ci"] = Z95 * means["std"] / np.sqrt(means["count"])
    return means


def ai(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    means = pair_type_means(model)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.1, 2.0))
    for k, (pair_type, (label, face, edge, marker)) in enumerate(PAIR_TYPE_STYLES.items()):
        rows = means[means.pair_type == pair_type].set_index("conf").loc[ML_VENUES]
        x = np.arange(len(ML_VENUES)) + (k - 1.5) * 0.17
        ax.errorbar(
            x, rows["mean"], yerr=rows.ci, fmt=marker, ms=3.5, mfc=face, mec=edge, mew=0.6,
            ecolor=edge, elinewidth=0.9, label=label,
        )
    ax.set_xlim(-0.5, len(ML_VENUES) - 0.5)
    ax.set_xticks(range(len(ML_VENUES)), rows.venue)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Critique overlap (%)")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2, handletextpad=0.3,
        handlelength=1.2, borderaxespad=0.2,
    )
    save(fig, figs, "fig2_ai.pdf", COLUMN_WIDTH)


def ai_checks(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    evidence = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"rq1_evidence_{model}.csv")
    fig = plt.figure(figsize=(PAGE_WIDTH - 0.15, 2.4))
    ax = fig.add_axes([0.37, 0.16, 0.61, 0.79])
    y = 0
    ticks, labels, is_header = [], [], []
    for group, rows in evidence.groupby("group", sort=False):
        ticks.append(y)
        labels.append(group)
        is_header.append(True)
        y -= 1
        placebo = group.startswith("Placebo")
        color = COLORS["grey"] if placebo else COLORS["ink"]
        face_color = "white" if placebo else color
        for row in rows.itertuples():
            ax.errorbar(row.est, y, xerr=Z95 * row.se, fmt="o", ms=3.5, color=color, mfc=face_color, elinewidth=1.0)
            ticks.append(y)
            labels.append(re.sub(r"(\d)-(\d)", "\\1–\\2", SHORT.get(row.label, row.label)))
            is_header.append(False)
            y -= 1
        y -= 0.4
    ax.axvline(0, color=COLORS["light"], lw=0.6, zorder=0)
    main_estimate = evidence.est.iloc[0]
    ax.axvline(main_estimate, color=COLORS["grey"], lw=0.6, ls=":", zorder=0)
    ax.set_yticks(ticks, labels)
    for tick, header in zip(ax.get_yticklabels(), is_header):
        tick.set_fontweight("bold" if header else "normal")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlim(-5, 12)
    few_ticks(ax, "x")
    ax.set_xlabel("Difference in overlap (percentage points)")
    save(fig, figs, "figA_checks.pdf", PAGE_WIDTH)


def spread_labels(ys, gap):
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed -= (placed.mean() - np.mean(ys)) / 2
    out = np.empty_like(placed)
    out[order] = placed
    return out


def overlap_panel(ax, by_year, styles, label_x, gap):
    ends = {}
    for conf, (name, color, marker, dx) in styles.items():
        rows = by_year[by_year.conf == conf]
        x, y, ci = rows.year + dx, rows["match"], Z95 * rows.match_se
        muted = color == COLORS["grey"]
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.1 if muted else 0.2, lw=0, zorder=1)
        ax.plot(x, y, "-", color=color, lw=0.9 if muted else 1.4, zorder=2 if muted else 3)
        ax.plot(x, y, marker, color=color, mec="white", mew=0.6, ms=3.5 if muted else 4, zorder=4)
        ends[name] = (x.iloc[-1], y.iloc[-1], color)
    label_ys = spread_labels([y for _, y, _ in ends.values()], gap)
    for (name, (x, y, color)), label_y in zip(ends.items(), label_ys):
        ax.plot([x, label_x - 0.08], [y, label_y], color=color, lw=0.5, alpha=0.6, zorder=2)
        ax.annotate(name, (label_x, label_y), ha="left", va="center", color=color, fontsize=7, annotation_clip=False)


def overlap(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    by_year = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"by_year_{model}.csv")
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(COLUMN_WIDTH - 0.1, 3.1))
    fig.subplots_adjust(left=0.13, right=0.76, top=0.93, bottom=0.11, hspace=0.42)
    conferences, journals = CONFERENCE_STYLES, JOURNAL_STYLES
    for ax, styles, title in [(top, conferences, "a  ML conferences"), (bottom, journals, "b  Journals")]:
        shade_chatgpt_period(ax)
        overlap_panel(ax, by_year, styles, 2026.65, 2.4)
        ax.set_ylim(5, 21)
        ax.set_yticks([5, 10, 15, 20])
        ax.set_ylabel("Critique overlap (%)")
        ax.set_title(title, fontweight="bold")
        ax.annotate(
            "LLM era",
            (CHATGPT_BAND[0], 21),
            xytext=(3, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=6.5,
            color=COLORS["grey"],
            style="italic",
        )
        ax.tick_params(axis="x", labelsize=6)
    bottom.set_xlim(2018.6, CHATGPT_BAND[1])
    bottom.set_xticks(range(2019, 2027))
    top.tick_params(labelbottom=True)
    bottom.set_xlabel("Review year")
    save(fig, figs, "fig1_trends.pdf", COLUMN_WIDTH)


def category_share_by_year(claims, year):
    counts = pd.crosstab(claims[claims.year == year].paper_id, claims.category[claims.year == year]) * 100
    claims_per_paper = counts.sum(axis=1) / 100
    share = counts.sum() / claims_per_paper.sum()
    residuals = counts - np.outer(claims_per_paper, share)
    n = len(counts)
    variance = (residuals**2).sum() / claims_per_paper.sum() ** 2 * n / (n - 1)
    return share, variance


def category_shifts(model):
    claims = pd.read_parquet(
        settings.DATA_DIR / f"claims_table_{model}.parquet", columns=["review_id", "paper_id", "conf", "year", "category"]
    )
    main_sample_reviews = main_sample_review_ids(settings.DATA_DIR)
    claims = claims[(claims.conf == "ICLR") & claims.review_id.isin(main_sample_reviews) & (claims.category != "other")]
    pre_years = list(range(ICLR_FROM, 2023))
    by_year = {year: category_share_by_year(claims, year) for year in pre_years + [2026]}
    pre = sum(by_year[year][0] for year in pre_years) / len(pre_years)
    pre_variance = sum(by_year[year][1] for year in pre_years) / len(pre_years) ** 2
    post, post_variance = by_year[2026]
    shift = pd.DataFrame({"pre": pre, "post": post, "change": post - pre, "se": np.sqrt(pre_variance + post_variance)})
    return shift.sort_values("change")


def categories(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    shift = category_shifts(model)
    fig, (levels, changes) = plt.subplots(
        1, 2, sharey=True, figsize=(COLUMN_WIDTH - 0.1, 2.45), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.1)
    )
    fig.subplots_adjust(left=0.37, right=0.97, top=0.92, bottom=0.15)
    y = np.arange(len(shift))
    levels.hlines(y, shift.pre, shift.post, color=COLORS["light"], lw=1.4, zorder=1)
    levels.plot(shift.post, y, "o", ms=3, color=COLORS["ink"], zorder=2)
    levels.plot(shift.pre, y, "o", ms=3, mfc="white", mew=0.8, color=COLORS["grey"], zorder=3)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=3, mfc="white", mew=0.8, color=COLORS["grey"]),
        plt.Line2D([], [], ls="", marker="o", ms=3, color=COLORS["ink"]),
    ]
    levels.legend(
        handles, [f"{ICLR_FROM}–22", "2026"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
        handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0,
    )
    levels.set_xlim(-1.5, 25)
    levels.set_xticks([0, 10, 20])
    levels.spines["bottom"].set_bounds(0, 20)
    levels.set_xlabel("Share of claims (%)")
    changes.axvline(0, color=COLORS["grey"], lw=0.6, zorder=0)
    changes.hlines(y, 0, shift.change, color=COLORS["light"], lw=1.4, zorder=1)
    changes.errorbar(shift.change, y, xerr=Z95 * shift.se, fmt="o", ms=3, color=COLORS["ink"], elinewidth=1.0, zorder=2)
    changes.set_xlim(-10.5, 6.5)
    changes.set_xticks([-10, -5, 0, 5])
    changes.spines["bottom"].set_bounds(-10, 5)
    changes.set_xlabel("Change (points)")
    changes.tick_params(axis="y", left=False)
    changes.spines["left"].set_visible(False)
    levels.set_yticks(y, [CATEGORY_NAMES.get(c, c) for c in shift.index])
    levels.tick_params(axis="y", length=0, pad=2)
    levels.spines["left"].set_visible(False)
    levels.set_ylim(-0.7, len(shift) - 0.3)
    save(fig, figs, "fig3_categories.pdf", COLUMN_WIDTH)


def journal_change(by_year, conf):
    rows = by_year[by_year.conf == conf]
    pre, last = rows[rows.year.between(2019, 2023)], rows[rows.year == rows.year.max()].iloc[0]
    return dict(
        overlap_change=last.match - pre.match.mean(),
        se=np.sqrt(last.match_se**2 + (pre.match_se**2).sum() / len(pre) ** 2),
        papers=last.papers + pre.papers.sum(),
    )


def fields(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    gaps = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"journal_gaps_{model}.csv")
    iclr = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"rq1_evidence_{model}.csv").iloc[0]
    sources = gaps.iloc[:-1].sort_values("est", ascending=False)
    pooled = gaps.iloc[-1]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.1, 1.9))
    fig.subplots_adjust(left=0.31, right=0.97, top=0.9, bottom=0.2)
    ax.axvspan(iclr.est - Z95 * iclr.se, iclr.est + Z95 * iclr.se, color="#D9E8F3", lw=0, zorder=0)
    ax.axvline(iclr.est, color=COLORS["ICLR"], lw=0.8, zorder=0)
    ax.axvline(0, color=COLORS["grey"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    rows = pd.concat([sources, pooled.to_frame().T])
    y = np.arange(len(rows) - 1, -1, -1)
    for yi, row in zip(y, rows.itertuples()):
        ax.errorbar(row.est, yi, xerr=Z95 * row.se, fmt="o", ms=3.5, color=COLORS["ink"], elinewidth=1.0, zorder=3)
    ax.axhline(0.55, color=COLORS["light"], lw=0.5)
    ax.annotate("ICLR", (iclr.est, len(sources) + 0.8), xytext=(0, 1), textcoords="offset points",
                fontsize=7, color=COLORS["ICLR"], ha="center", va="bottom", annotation_clip=False)
    labels = [f"{JOURNALS[r.source]} ({r.both_papers:,})" for r in sources.itertuples()]
    ax.set_yticks(y, [*labels, f"All journals ({int(pooled.both_papers):,})"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(sources) + 0.8)
    ax.set_xlim(-12, 20)
    ax.set_xticks([-10, 0, 10, 20])
    ax.spines["bottom"].set_bounds(-10, 20)
    ax.set_xlabel("Difference in overlap (points)")
    save(fig, figs, "fig4_fields.pdf", COLUMN_WIDTH)


def cross(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    placebo = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"placebo_outcomes_{model}.parquet"))
    other = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"other_panel_outcomes_{model}.parquet"))
    venue = placebo[placebo.placebo_kind == SAME_VENUE_PLACEBO]
    iclr = placebo[placebo.conf == "ICLR"]
    fig = plt.figure(figsize=(PAGE_WIDTH - 0.1, 3.0))
    grid = fig.add_gridspec(2, 10, left=0.07, right=0.92, top=0.93, bottom=0.08, hspace=0.75, wspace=1.6)
    venues_ax, area_ax = fig.add_subplot(grid[0, :5]), fig.add_subplot(grid[0, 5:])
    journal_axes = [fig.add_subplot(grid[1, 2 * k : 2 * k + 2]) for k in range(len(JOURNALS))]

    iclr_venue = mean_line(venues_ax, venue[venue.conf == "ICLR"], "hit_rate_strict", COLORS["ICLR"])
    neurips_venue = mean_line(venues_ax, venue[venue.conf == "NeurIPS"], "hit_rate_strict", COLORS["NeurIPS"])
    end_label(venues_ax, 2026, iclr_venue.loc[2026, "mean"], "ICLR", COLORS["ICLR"], dx=5)
    end_label(venues_ax, 2025, neurips_venue.loc[2025, "mean"], "NeurIPS", COLORS["NeurIPS"], 0, -8, "center", "top")
    venues_ax.set_title("a  Different papers, one venue-year", fontweight="bold")

    same_venue = mean_line(area_ax, iclr[iclr.placebo_kind == SAME_VENUE_PLACEBO], "hit_rate_strict", COLORS["ink"])
    area_rows = iclr[iclr.placebo_kind == "same_area3"]
    same_area = mean_line(area_ax, area_rows, "hit_rate_strict", COLORS["ink"], ls="--", mfc="white")
    end_label(area_ax, 2026, same_area.loc[2026, "mean"], "same area", dx=5)
    end_label(area_ax, 2026, same_venue.loc[2026, "mean"], "same venue", dx=5)
    area_ax.set_title("b  ICLR: same venue-year vs. same research area", fontweight="bold")

    for ax, (conf, title) in zip(journal_axes, JOURNALS.items(), strict=True):
        mean_line(ax, other[(other.conf == conf) & other.placebo], "hit_rate_strict", COLORS["ink"])
        ax.set_title(title)
    fig.text(0.01, 0.47, "c  Journals: different papers, one source-year", fontweight="bold", fontsize=8, ha="left")

    for ax in [venues_ax, area_ax, *journal_axes]:
        shade_chatgpt_period(ax)
        ax.set_ylim(-0.3, 4)
        ax.set_yticks([0, 2, 4])
    for ax in [venues_ax, area_ax]:
        ax.set_xlim(2019.6, 2026.4)
        year_ticks(ax)
    for ax in journal_axes:
        ax.set_xlim(2018.4, 2026.6)
        year_ticks(ax, (2019, 2022, 2026))
    for ax in [venues_ax, journal_axes[0]]:
        ax.set_ylabel("Overlap (%)")
    save(fig, figs, "figA_cross.pdf", PAGE_WIDTH)


def embedding(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    embed = in_window(pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"embed_{model}.csv"))
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.5, 1.8))
    shade_chatgpt_period(ax)
    by_conf = {conf: embed[embed.conf == conf].set_index("year") for conf in ["ICLR", "NeurIPS"]}
    for conf, rows in by_conf.items():
        ax.plot(rows.index, rows.intra, "o-", ms=3, color=COLORS[conf])
        ax.plot(rows.index, rows.inter_area, "o--", ms=3, mfc="white", color=COLORS[conf])
    iclr, neurips = by_conf["ICLR"], by_conf["NeurIPS"]
    end_label(ax, 2026, iclr.intra.iloc[-1], "ICLR, same paper", COLORS["ICLR"])
    end_label(ax, 2025, neurips.intra.iloc[-1], "NeurIPS, same paper", COLORS["NeurIPS"], dy=6, ha="right", dx=0)
    end_label(ax, 2026, iclr.inter_area.iloc[-1], "ICLR, different\npapers, same area", COLORS["ICLR"])
    end_label(ax, 2023, neurips.inter_area.loc[2023], "NeurIPS, different papers", COLORS["NeurIPS"], dx=0, dy=-10, ha="center")
    ax.set_xlim(2019.6, 2026.4)
    ax.set_ylim(0, 0.9)
    year_ticks(ax)
    ax.set_ylabel("Cosine similarity")
    save(fig, figs, "figA_embedding.pdf", COLUMN_WIDTH)


def all_figures(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    for draw in [overlap, ai, ai_checks, categories, fields, cross, embedding, rates]:
        draw(figs, model)


def summarize(df, placebo):
    groups = df.groupby(["conf", "year"])
    out = pd.DataFrame(
        {
            "papers": groups.size(),
            "rate_papers": groups.hit_rate_strict.count(),
            "reviews": groups.n_reviews.sum(),
            "claims_per_review": groups.claims_per_review.mean(),
            "match": groups.hit_rate_strict.mean() * 100,
            "match_se": groups.hit_rate_strict.sem() * 100,
        }
    )
    return out.join(placebo.groupby(["conf", "year"]).hit_rate_strict.mean().mul(100).rename("match_other_papers"))


def by_year(model=settings.DEFAULT_MODEL):
    ml = pd.read_parquet(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet")
    placebo = pd.read_parquet(settings.ANALYSIS_DIR / f"placebo_outcomes_{model}.parquet")
    ml_placebo = placebo.query("placebo_kind == @SAME_VENUE_PLACEBO")
    icml = pd.read_parquet(settings.ANALYSIS_DIR / f"icml_panel_outcomes_{model}.parquet")
    other = pd.read_parquet(settings.ANALYSIS_DIR / f"other_panel_outcomes_{model}.parquet")
    rows = pd.concat(
        [summarize(ml, ml_placebo), summarize(icml, icml[icml.placebo]), summarize(other[~other.placebo], other[other.placebo])]
    )

    flags = meld_flags()
    flag_rates = flags.groupby(["conf", "year"])[["flag", "written"]].mean().mul(100)
    rows = rows.join(flag_rates.rename(columns={"flag": "ai_flagged", "written": "ai_written"}))

    rows = in_window(rows.reset_index())
    rows["conf"] = pd.Categorical(rows.conf, ORDER)
    return rows.sort_values(["conf", "year"])


def tex_number(v):
    return "--" if pd.isna(v) else f"{v:.1f}"


def latex(rows):
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Source & Year & Papers & Reviews & Claims/rev. & Overlap (\%) & Other papers (\%) & AI-flagged (\%) & AI-written (\%) \\",
        r"\midrule",
    ]
    for conf, group in rows.groupby("conf", observed=True):
        for k, row in enumerate(group.itertuples()):
            cells = [
                str(conf) if k == 0 else "",
                str(row.year),
                f"{row.papers:,}",
                f"{row.reviews:,}",
                tex_number(row.claims_per_review),
                f"{tex_number(row.match)} $\\pm$ {tex_number(row.match_se)}",
                tex_number(row.match_other_papers),
                tex_number(row.ai_flagged),
                tex_number(row.ai_written),
            ]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def table(csv=None, tables=settings.TABLES_DIR, model=settings.DEFAULT_MODEL):
    rows = by_year(model)
    rows.to_csv(csv or settings.ANALYSIS_REPORTS_DIR / f"by_year_{model}.csv", index=False)
    print(rows.round(2).to_string(index=False))
    is_ml = rows.conf.isin(ML_VENUES)
    Path(tables, "by_year_ml.tex").write_text(latex(rows[is_ml]))
    Path(tables, "by_year_journals.tex").write_text(latex(rows[~is_ml]))
    lines = [r"\begin{tabular}{llrr}", r"\toprule", r"Source & Years & Papers & Reviews \\", r"\midrule"]
    for conf, source in rows.groupby("conf", observed=True):
        years = f"{source.year.min()}--{source.year.max() % 100:02d}"
        lines.append(f"{JOURNALS.get(conf, conf)} & {years} & {source.papers.sum():,} & {source.reviews.sum():,} \\\\")
    Path(tables, "corpus.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def numbers(model=settings.DEFAULT_MODEL, tables=settings.TABLES_DIR):
    reports = settings.ANALYSIS_REPORTS_DIR
    by_year = pd.read_csv(reports / f"by_year_{model}.csv")
    evidence = pd.read_csv(reports / f"rq1_evidence_{model}.csv").set_index("label")
    values, rows = {}, []

    def record(name, estimate, error=None, n=None, digits=1):
        values[name] = f"{estimate:.{digits}f}"
        if error is not None:
            values[name + "SE"] = f"{error:.{digits}f}"
            values[name + "Low"] = f"{estimate - Z95 * error:.{digits}f}"
            values[name + "High"] = f"{estimate + Z95 * error:.{digits}f}"
        if n is not None:
            values[name + "N"] = f"{int(n):,}"
        rows.append(dict(name=name, estimate=estimate, se=error, n=n))

    for conf, prefix, base_year, last_year in [
        ("ICLR", "Iclr", 2023, 2026),
        ("NeurIPS", "Neurips", 2024, 2025),
        ("ICML", "Icml", 2025, 2026),
    ]:
        source = by_year[by_year.conf == conf].set_index("year")
        base, last = source.loc[base_year], source.loc[last_year]
        record(prefix + "Base", base.match, base.match_se, base.rate_papers)
        record(prefix + "Last", last.match, last.match_se, last.rate_papers)
        record(
            prefix + "Change",
            last.match - base.match,
            np.hypot(last.match_se, base.match_se),
            base.rate_papers + last.rate_papers,
        )
    forms = pd.read_csv(reports / f"forms_{model}.csv").set_index("year")
    for year, suffix in [(2022, "Prev"), (2023, "Base"), (2024, "Next")]:
        row = forms.loc[year]
        record("FormWithout" + suffix, row.hit_without, row.hit_without_se, row.papers)
        record("FormShared" + suffix, row.form_share, row.form_share_se, row.papers, digits=0)
    for year, prefix in [(2023, "FieldRemoved"), (2024, "QuestionsRemoved")]:
        row = forms.loc[year]
        record(prefix, row.hit_removed, row.hit_removed_se, row.removed_papers)
        record(prefix + "Change", row.removed_change, row.removed_change_se, row.removed_papers)
    for label, prefix in [
        ("ICLR 2024–26", "FlagGap"),
        ("+ length and claim count", "ControlGap"),
        ("NeurIPS 2024–25", "NeuripsFlagGap"),
        ("ICML 2025–26", "IcmlFlagGap"),
        ("AI-written, ICLR", "WrittenGap"),
    ]:
        row = evidence.loc[label]
        record(prefix, row.est, row.se, row.papers)
    gaps = pd.read_csv(reports / f"journal_gaps_{model}.csv").set_index("source")
    for source, prefix in JOURNAL_PREFIX.items():
        row = gaps.loc[source]
        record(prefix + "Gap", row.est, row.se, row.papers)
    pooled = gaps.loc["All journals"]
    record("JournalGap", pooled.est, pooled.se, pooled.papers)
    values["JournalGapBothN"] = f"{int(pooled.both_papers):,}"
    meld = pd.read_csv(settings.REPORTS_DIR / "meld_summary.csv").set_index("name").value
    values["MeldThreshold"] = f"{meld.threshold:.2f}"
    for name in ["calibration_reviews", "intel_ai_reviews", "rewrite_reviews"]:
        values["Meld" + name.title().replace("_", "")] = f"{int(meld[name]):,}"
    for name in ["pre_source_max", "pre_year_max", "intel_ai_recall", "rewrite_recall_min"]:
        values["Meld" + name.title().replace("_", "")] = f"{meld[name]:.1f}"
    iclr_last = by_year[(by_year.conf == "ICLR") & (by_year.year == 2026)].iloc[0]
    record("IclrFlaggedLast", iclr_last.ai_flagged)
    record("IclrWrittenLast", iclr_last.ai_written)
    flags = meld_flags().set_index("review_id").flag
    pairs = pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet")
    pairs["nflag"] = pairs.a.map(flags).astype(float) + pairs.b.map(flags).astype(float)
    baseline = pairs[(pairs.conf == "ICLR") & (pairs.year >= 2024) & (pairs.nflag == 0)].groupby("panel_id").hit.mean() * 100
    record("FlagBaseline", baseline.mean(), baseline.sem(), len(baseline))
    record("FlagRelative", evidence.loc["ICLR 2024–26"].est / baseline.mean() * 100, digits=0)
    written = pair_type_means(model).query("pair_type == 'WW'").set_index("conf")["count"]
    for conf in ML_VENUES:
        values[f"PairWritten{conf.title()}N"] = f"{int(written[conf]):,}"
    placebo = pd.read_parquet(settings.ANALYSIS_DIR / f"placebo_outcomes_{model}.parquet")
    for year, prefix in [(2023, "CrossBase"), (2026, "CrossLast")]:
        sample = (
            placebo[
                (placebo.conf == "ICLR") & (placebo.year == year) & (placebo.placebo_kind == SAME_VENUE_PLACEBO)
            ].hit_rate_strict.dropna()
            * 100
        )
        record(prefix, sample.mean(), sample.sem(), len(sample))
    embed = pd.read_csv(reports / f"embed_{model}.csv").set_index(["conf", "year"])
    for year, suffix in [(2020, "Base"), (2023, "Mid"), (2026, "Last")]:
        row = embed.loc[("ICLR", year)]
        record("EmbedSame" + suffix, row.intra, row.intra_se, row.papers, digits=2)
        record("EmbedArea" + suffix, row.inter_area, digits=2)
    categories = category_shifts(model)
    categories.to_csv(reports / f"category_shift_{model}.csv")
    values["CategoryMaxInterval"] = f"{(Z95 * categories.se).max():.1f}"
    for category, prefix in [
        ("clarity/presentation", "Clarity"),
        ("typo/formatting", "Typos"),
        ("experiments-scope", "Experiments"),
    ]:
        row = categories.loc[category]
        record(prefix + "Base", row.pre, digits=0)
        record(prefix + "Last", row.post, digits=0)
    shifts = []
    for conf, prefix in JOURNAL_PREFIX.items():
        change = journal_change(by_year, conf)
        record(prefix + "Change", change["overlap_change"], change["se"], change["papers"])
        shifts.append(dict(source=conf, **change))
    pd.DataFrame(shifts).to_csv(reports / f"journal_changes_{model}.csv", index=False)
    ref = pd.read_csv(reports / f"reference_{model}.csv").set_index(["kind", "years", "split"])
    same, cross = ref.loc[("same", "all", "all")], ref.loc[("cross", "all", "all")]
    change = ref.loc[("same", "2026 minus earlier", "all")]
    values["RefPanels"] = f"{int(same.panels)}"
    values["RefCrossPanels"] = f"{int(cross.panels)}"
    values["RefReviews"] = f"{int(same.reviews + cross.reviews)}"
    values["RefMatchedReviews"] = f"{int(same.matched_reviews + cross.matched_reviews)}"
    values["RefOverlap"] = f"{same.overlap_ref:.1f}"
    values["RefOverlapLuna"] = f"{same.overlap_model:.1f}"
    values["RefOverlapRho"] = f"{same.overlap_rho:.2f}"
    values["RefCrossLuna"] = f"{cross.overlap_model:.1f}"
    values["RefClaims"] = f"{same.claims_ref:.1f}"
    values["RefClaimsLuna"] = f"{same.claims_model:.1f}"
    for row, prefix in [(same, "Ref"), (change, "RefChange")]:
        for name, digits in [("claims_bias", 2), ("overlap_bias", 1), ("recall", 1), ("precision", 1)]:
            for suffix in ("", "_low", "_high"):
                values[prefix + name.title().replace("_", "") + suffix.title()[1:]] = f"{row[name + suffix]:.{digits}f}"
    pd.DataFrame(rows).to_csv(reports / f"paper_numbers_{model}.csv", index=False)
    Path(tables, "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()) + "\n"
    )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    fire.Fire(
        {
            "all": all_figures,
            "ai": ai,
            "ai_checks": ai_checks,
            "overlap": overlap,
            "categories": categories,
            "fields": fields,
            "embedding": embedding,
            "rates": rates,
            "cross": cross,
            "table": table,
            "numbers": numbers,
        }
    )
