import re
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

import extract
import hub
from outcomes import analysis_pairs, excluded_review_ids, in_window, main_sample_review_ids
from prompts import render
from settings import settings

COLUMN_WIDTH = 3.03
PAGE_WIDTH = 6.3
Z95 = 1.96
LABEL_SIZE = 9
REWRITE_MACROS = [
    ("outline", "Rw", "rewrite_panel"),
    ("polish", "Pl", "polish_panel"),
    ("outline_second", "Rk", "rewrite_panel_second"),
    ("polish_second", "Pk", "polish_panel_second"),
]
TEXT_DARKEN = 0.35
CHATGPT_BAND = (2022.6, settings.ICLR_LAST + 0.5)
SAME_VENUE_PLACEBO = "shuffled4"
CATEGORY_YEAR = 2026
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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
JOURNALS = {"SciPost": "SciPost", "F1000": "F1000", "PLOS": "PLOS", "BMC-Med": "BMC", "Copernicus": "Copernicus"}
ORDER = settings.ML_VENUES + list(JOURNALS)
JOURNAL_PREFIX = {"F1000": "FResearch", "PLOS": "Plos", "SciPost": "Scipost", "BMC-Med": "Bmc", "Copernicus": "Copernicus"}
SHORT = {
    "+ length and claim count": "+ length, claim count",
    "99th-percentile threshold, ICLR": "99th-pct. threshold",
    "99.5th-percentile threshold, ICLR": "99.5th-pct. threshold",
    "Top-45% scores, ICLR 2020": "ICLR 2020",
    "Top-45% scores, ICLR 2021-2022": "ICLR 2021–22",
    "Top-45% scores, ICLR 2023": "ICLR 2023",
}
GAP_PREFIX = dict(zip(settings.GAP_CELLS, ["Iclr", "IclrLate", "Neurips", "Icml"], strict=True))
CHECK_GROUPS = {
    "Pipeline": "Main estimate and variants",
    "ICLR by year": "ICLR by year",
    "Placebo before ChatGPT": "Before ChatGPT, top 45% of scores",
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
    "BMC-Med": ("BMC", COLORS["grey"], "D", 0),
}


def shade_chatgpt_period(ax):
    ax.axvspan(*CHATGPT_BAND, color=COLORS["band"], zorder=0, lw=0)


def text_color(color):
    return tuple(c * (1 - TEXT_DARKEN) for c in to_rgb(color))


def end_label(ax, x, y, text, color=COLORS["ink"], dx=3, dy=0, ha="left", va="center"):
    ax.annotate(
        text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=text_color(color), fontsize=LABEL_SIZE
    )


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


def kept_meld_flags():
    flags = hub.meld_flags(["conf", "year", "flag"])
    return flags[~flags.index.isin(excluded_review_ids())]


PAIR_TYPE_STYLES = {
    "HH": ("Both unflagged", COLORS["grey"], COLORS["grey"], "o"),
    "HL": ("One flagged", COLORS["flag_light"], COLORS["flag"], "s"),
    "LL": ("Both flagged", COLORS["flag"], COLORS["flag"], "D"),
}


def pair_type_means(model):
    flags = kept_meld_flags()
    pairs = analysis_pairs(model)
    cells = []
    for label, (conf, first, last) in settings.GAP_CELLS.items():
        cells.append(pairs[(pairs.conf == conf) & pairs.year.between(first, last)].assign(conf=label))
    pairs = pd.concat(cells)
    n_flagged = pairs.a.map(flags.flag).astype(int) + pairs.b.map(flags.flag).astype(int)
    pairs["pair_type"] = n_flagged.map({0: "HH", 1: "HL", 2: "LL"})
    per_paper = pairs.groupby(["conf", "pair_type", "panel_id"]).hit.mean() * 100
    means = per_paper.groupby(["conf", "pair_type"]).agg(["mean", "std", "count"]).reset_index()
    means["ci"] = Z95 * means["std"] / np.sqrt(means["count"])
    return means


def ai(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    means = pair_type_means(model)
    contrasts = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"pair_types_{model}.csv")
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(COLUMN_WIDTH - 0.35, 3.4), gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.38)
    )
    cells = list(settings.GAP_CELLS)
    for k, (pair_type, (label, face, edge, marker)) in enumerate(PAIR_TYPE_STYLES.items()):
        x = np.arange(len(cells)) + (k - 1) * 0.2
        style = dict(fmt=marker, ms=3.5, mfc=face, mec=edge, mew=0.6, ecolor=edge, elinewidth=0.9)
        rows = means[means.pair_type == pair_type].set_index("conf").loc[cells]
        top.errorbar(x, rows["mean"], yerr=rows.ci, label=label, **style)
        if pair_type == "HH":
            bottom.errorbar(x, np.zeros(len(x)), **style)
            continue
        cell = contrasts[contrasts.pair_type == pair_type].set_index("conf").loc[cells]
        bottom.errorbar(x, cell.est, yerr=Z95 * cell.se, **style)
    bottom.axhline(0, color=COLORS["light"], lw=0.6, zorder=0)
    bottom.set_xlim(-0.5, len(cells) - 0.5)
    bottom.set_xticks(range(len(cells)), [label.replace(" ", "\n") for label in cells])
    bottom.tick_params(axis="x", length=0)
    top.tick_params(axis="x", length=0)
    top.set_ylim(0, 30)
    top.set_yticks([0, 10, 20, 30])
    top.set_ylabel("Overlap (%)")
    top.set_title("a  Mean overlap by pair type", fontweight="bold")
    bottom.set_ylim(-1.5, 16)
    bottom.set_yticks([0, 5, 10, 15])
    bottom.set_ylabel("Difference (points)")
    bottom.set_title("b  Difference from both unflagged", fontweight="bold")
    fig.align_ylabels([top, bottom])
    top.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        columnspacing=0.5,
        handletextpad=0.2,
        handlelength=1.0,
        borderaxespad=0.2,
    )
    save(fig, figs, "fig2_ai.pdf", COLUMN_WIDTH)


def ai_checks(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    evidence = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"rq1_evidence_{model}.csv")
    fig = plt.figure(figsize=(PAGE_WIDTH - 0.15, 3.7))
    ax = fig.add_axes([0.33, 0.12, 0.45, 0.86])
    sides = ax.get_yaxis_transform()
    y = 0
    for group, rows in evidence.groupby("group", sort=False):
        ax.text(-0.7, y, CHECK_GROUPS[group], transform=sides, ha="left", va="center", fontweight="bold")
        y -= 1
        before = group.startswith("Placebo")
        color = COLORS["grey"] if before else COLORS["ink"]
        for row in rows.itertuples():
            low, high = row.est - Z95 * row.se, row.est + Z95 * row.se
            ax.errorbar(
                row.est, y, xerr=Z95 * row.se, fmt="o", ms=3.5, color=color, mfc="white" if before else color, elinewidth=1.0
            )
            label = re.sub(r"(\d)-(\d)", "\\1–\\2", SHORT.get(row.label, row.label))
            ax.text(-0.66, y, label, transform=sides, ha="left", va="center")
            values = f"{row.est:+.1f}  [{low:+.1f}, {high:+.1f}]".replace("-", "−")
            ax.text(1.04, y, values, transform=sides, ha="left", va="center", color=text_color(color))
            y -= 1
        y -= 0.5
    ax.text(1.04, 1, "Points  [95% CI]", transform=sides, ha="left", va="center", fontweight="bold")
    ax.axvline(0, color=COLORS["light"], lw=0.6, zorder=0)
    ax.axvline(evidence.est.iloc[0], color=COLORS["grey"], lw=0.6, ls=":", zorder=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.9, 1.6)
    ax.set_xlim(-10, 22)
    ax.set_xticks([-5, 0, 5, 10, 15])
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
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.12 if muted else 0.2, lw=0, zorder=1)
        width, order = (1.1, 2) if muted else (1.4, 3)
        final = int((rows.year < settings.ICLR_SNAPSHOT_YEARS[0]).sum()) if conf == "ICLR" else len(x)
        ax.plot(x[:final], y[:final], "-", color=color, lw=width, zorder=order)
        ax.plot(x[final - 1 : final + 1], y[final - 1 : final + 1], "--", color=color, lw=width, dashes=(2.2, 1.6), zorder=order)
        ax.plot(x[final:], y[final:], "-", color=color, lw=width, zorder=order)
        ax.plot(x, y, marker, color=color, mec="white", mew=0.6, ms=3.5 if muted else 4, zorder=4)
        ends[name] = (x.iloc[-1], y.iloc[-1], color)
    label_ys = spread_labels([y for _, y, _ in ends.values()], gap)
    for (name, (x, y, color)), label_y in zip(ends.items(), label_ys):
        ax.plot([x, label_x - 0.08], [y, label_y], color=color, lw=0.5, alpha=0.8, zorder=2)
        ax.annotate(
            name, (label_x, label_y), ha="left", va="center", color=text_color(color), fontsize=LABEL_SIZE, annotation_clip=False
        )


def overlap(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    by_year = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"by_year_{model}.csv")
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(COLUMN_WIDTH - 0.1, 3.1))
    fig.subplots_adjust(left=0.13, right=0.76, top=0.93, bottom=0.11, hspace=0.42)
    conferences, journals = CONFERENCE_STYLES, JOURNAL_STYLES
    for ax, styles, title in [(top, conferences, "a  ML conferences"), (bottom, journals, "b  Journals")]:
        shade_chatgpt_period(ax)
        overlap_panel(ax, by_year, styles, settings.ICLR_LAST + 0.65, 2.6)
        ax.set_ylim(0, 27)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_ylabel("Critique overlap (%)")
        ax.set_title(title, fontweight="bold")
        ax.annotate(
            "After ChatGPT",
            (CHATGPT_BAND[0], 27),
            xytext=(3, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=COLORS["grey"],
            style="italic",
        )
    bottom.set_xlim(2018.6, CHATGPT_BAND[1])
    bottom.set_xticks(range(2019, settings.ICLR_LAST + 1), [f"'{year % 100}" for year in range(2019, settings.ICLR_LAST + 1)])
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
        settings.WORK_DIR / f"claims_table_{model}.parquet", columns=["review_id", "paper_id", "conf", "year", "category"]
    )
    main_sample_reviews = main_sample_review_ids()
    claims = claims[(claims.conf == "ICLR") & claims.review_id.isin(main_sample_reviews) & (claims.category != "other")]
    pre_years = list(range(settings.ICLR_FROM, 2023))
    by_year = {year: category_share_by_year(claims, year) for year in pre_years + [CATEGORY_YEAR]}
    pre = sum(by_year[year][0] for year in pre_years) / len(pre_years)
    pre_variance = sum(by_year[year][1] for year in pre_years) / len(pre_years) ** 2
    post, post_variance = by_year[CATEGORY_YEAR]
    shift = pd.DataFrame({"pre": pre, "post": post, "change": post - pre, "se": np.sqrt(pre_variance + post_variance)})
    return shift.sort_values("change")


def categories(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    shift = category_shifts(model)
    fig, (levels, changes) = plt.subplots(
        1, 2, sharey=True, figsize=(PAGE_WIDTH - 0.2, 2.8), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.1)
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.93, bottom=0.14)
    y = np.arange(len(shift))
    levels.hlines(y, shift.pre, shift.post, color=COLORS["light"], lw=1.4, zorder=1)
    levels.plot(shift.post, y, "o", ms=3, color=COLORS["ink"], zorder=2)
    levels.plot(shift.pre, y, "o", ms=3, mfc="white", mew=0.8, color=COLORS["grey"], zorder=3)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=3, mfc="white", mew=0.8, color=COLORS["grey"]),
        plt.Line2D([], [], ls="", marker="o", ms=3, color=COLORS["ink"]),
    ]
    levels.legend(
        handles,
        [f"{settings.ICLR_FROM}–22", str(CATEGORY_YEAR)],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    levels.set_xlim(-1.5, 25)
    levels.set_xticks([0, 10, 20])
    levels.spines["bottom"].set_bounds(0, 20)
    levels.set_xlabel("Share (%)")
    changes.axvline(0, color=COLORS["grey"], lw=0.6, zorder=0)
    changes.hlines(y, 0, shift.change, color=COLORS["light"], lw=1.4, zorder=1)
    changes.errorbar(shift.change, y, xerr=Z95 * shift.se, fmt="o", ms=3, color=COLORS["ink"], elinewidth=1.0, zorder=2)
    changes.set_xlim(-12.5, 6.5)
    changes.set_xticks([-10, 0, 5])
    changes.spines["bottom"].set_bounds(-10, 5)
    changes.set_xlabel("Change (points)")
    changes.tick_params(axis="y", left=False)
    changes.spines["left"].set_visible(False)
    levels.set_yticks(y, [CATEGORY_NAMES.get(c, c) for c in shift.index])
    levels.tick_params(axis="y", length=0, pad=2)
    levels.spines["left"].set_visible(False)
    levels.set_ylim(-0.7, len(shift) - 0.3)
    save(fig, figs, "fig3_categories.pdf", PAGE_WIDTH)


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
    identifying = gaps.set_index("source").varying
    sources = gaps.iloc[:-1].dropna(subset="se").sort_values("est", ascending=False)
    pooled = gaps.iloc[-1]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.35, 1.75))
    fig.subplots_adjust(left=0.31, right=0.97, top=0.9, bottom=0.2)
    ax.axvspan(iclr.est - Z95 * iclr.se, iclr.est + Z95 * iclr.se, color="#D9E8F3", lw=0, zorder=0)
    ax.axvline(iclr.est, color=COLORS["ICLR"], lw=0.8, zorder=0)
    ax.axvline(0, color=COLORS["grey"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    rows = pd.concat([sources, pooled.to_frame().T])
    y = np.arange(len(rows) - 1, -1, -1)
    for yi, row in zip(y, rows.itertuples()):
        ax.errorbar(row.est, yi, xerr=Z95 * row.se, fmt="o", ms=3.5, color=COLORS["ink"], elinewidth=1.0, zorder=3)
    ax.axhline(0.55, color=COLORS["light"], lw=0.5)
    ax.annotate(
        "ICLR",
        (iclr.est, len(sources) + 0.8),
        xytext=(0, 1),
        textcoords="offset points",
        fontsize=8,
        color=COLORS["ICLR"],
        ha="center",
        va="bottom",
        annotation_clip=False,
    )
    labels = [f"{JOURNALS[r.source]} ({identifying[r.source]:,})" for r in sources.itertuples()]
    ax.set_yticks(y, [*labels, f"All journals ({identifying['All journals']:,})"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(sources) + 0.8)
    ax.set_xlim(-12, 20)
    ax.set_xticks([-10, 0, 10, 20])
    ax.spines["bottom"].set_bounds(-10, 20)
    ax.set_xlabel("Difference in overlap (points)")
    save(fig, figs, "fig4_fields.pdf", COLUMN_WIDTH)


def calibrate_embedding(cells):
    cells = cells.copy()
    baseline = cells[cells.year.between(settings.ICLR_FROM, 2022)].groupby("conf").inter_area.mean()
    cells["calibration_baseline"] = cells.conf.map(baseline)
    for column in ["intra", "inter_area"]:
        cells["calibrated_" + column] = (cells[column] - cells.calibration_baseline) / (1 - cells.calibration_baseline)
    cells["calibrated_intra_se"] = cells.intra_se / (1 - cells.calibration_baseline)
    return cells


def embedding(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    embed = calibrate_embedding(in_window(pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"embed_{model}.csv")))
    embed.to_csv(settings.ANALYSIS_REPORTS_DIR / f"embed_calibrated_{model}.csv", index=False)
    fig, axes = plt.subplots(
        1, 2, sharey=True, figsize=(PAGE_WIDTH - 0.1, 2.3), gridspec_kw=dict(width_ratios=[8, 5], wspace=0.12)
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12)
    for ax, conf, letter in zip(axes, ["ICLR", "NeurIPS"], "ab"):
        rows = embed[embed.conf == conf].set_index("year")
        color = COLORS[conf]
        shade_chatgpt_period(ax)
        ax.axhline(0, color=COLORS["grey"], lw=0.6, zorder=0)
        ax.errorbar(rows.index, rows.calibrated_intra, yerr=Z95 * rows.calibrated_intra_se, fmt="o-", ms=3, color=color)
        ax.plot(rows.index, rows.calibrated_inter_area, "o--", ms=3, mfc="white", color=color)
        end_label(ax, rows.index[-1], rows.calibrated_intra.iloc[-1], "same paper", color, dx=0, dy=-12, ha="right", va="top")
        end_label(
            ax, rows.index[-1], rows.calibrated_inter_area.iloc[-1], "different papers, one area", color, 0, 7, "right", "bottom"
        )
        ax.set_title(f"{letter}  {conf}", fontweight="bold")
        ax.set_xlim(rows.index[0] - 0.4, rows.index[-1] + 0.4)
        ax.set_xticks(list(rows.index), [f"'{year % 100}" for year in rows.index])
    axes[0].set_ylim(-0.18, 0.75)
    axes[0].set_yticks([0, 0.3, 0.6])
    axes[0].set_ylabel("Baseline-scaled similarity")
    save(fig, figs, "figA_embedding.pdf", PAGE_WIDTH)


def all_figures(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    for draw in [overlap, ai, ai_checks, categories, fields, embedding]:
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

    flags = kept_meld_flags()
    flag_rates = flags.groupby(["conf", "year"]).flag.mean().mul(100)
    rows = rows.join(flag_rates.rename("ai_flagged"))

    rows = in_window(rows.reset_index())
    rows["conf"] = pd.Categorical(rows.conf, ORDER)
    return rows.sort_values(["conf", "year"])


def tex_number(v):
    return "--" if pd.isna(v) else f"{v:.1f}"


JOURNAL_PERIODS = ((2019, 2023), (2024, 2026))


def journal_periods(rows):
    out = []
    for conf, source in rows[~rows.conf.isin(settings.ML_VENUES)].groupby("conf", observed=True):
        for first, last in JOURNAL_PERIODS:
            cell = source[source.year.between(first, last)]
            by_reviews = cell.reviews / cell.reviews.sum()
            by_papers = cell.papers / cell.papers.sum()
            out.append(
                dict(
                    conf=conf,
                    year=f"{first}--{cell.year.max() % 100}",
                    papers=cell.papers.sum(),
                    reviews=cell.reviews.sum(),
                    claims_per_review=(cell.claims_per_review * by_reviews).sum(),
                    match=(cell.match * by_papers).sum(),
                    match_se=np.sqrt(((cell.match_se * by_papers) ** 2).sum()),
                    ai_flagged=(cell.ai_flagged * by_reviews).sum(),
                )
            )
    return pd.DataFrame(out).assign(conf=lambda d: pd.Categorical(d.conf, ORDER))


def latex(rows):
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Source & Year & Papers & Reviews & \shortstack[r]{Claims\\per review} & \shortstack[r]{Overlap\\(\%)} & \shortstack[r]{LLM-flagged\\(\%)} \\",
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
                tex_number(row.ai_flagged),
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
    Path(tables, "by_year.tex").write_text(latex(pd.concat([rows[rows.conf.isin(settings.ML_VENUES)], journal_periods(rows)])))
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
        ("ICLR", "Iclr", 2023, settings.ICLR_LAST),
        ("ICLR", "IclrRecent", 2025, 2026),
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
        ("ICLR 2024–27", "FlagGap"),
        ("+ length and claim count", "ControlGap"),
        ("ICLR 2024", "FlagGapTwentyFour"),
        ("ICLR 2027", "FlagGapTwentySeven"),
    ]:
        row = evidence.loc[label]
        record(prefix, row.est, row.se, row.papers)
        values[prefix + "Ident"] = f"{int(row.identifying):,}"
    for label, name in [
        ("99th-percentile threshold, ICLR", "ThresholdLowGap"),
        ("99.5th-percentile threshold, ICLR", "ThresholdMidGap"),
        ("Top-45% scores, ICLR 2020", "PlaceboFirstGap"),
        ("Top-45% scores, ICLR 2021-2022", "PlaceboMidGap"),
        ("Top-45% scores, ICLR 2023", "PlaceboLastGap"),
    ]:
        record(name, evidence.loc[label].est, evidence.loc[label].se)
    gaps = pd.read_csv(reports / f"journal_gaps_{model}.csv").set_index("source")
    for source, prefix in JOURNAL_PREFIX.items():
        row = gaps.loc[source]
        record(prefix + "Gap", row.est, row.se, row.papers)
    pooled = gaps.loc["All journals"]
    record("JournalGap", pooled.est, pooled.se, pooled.papers)
    values["JournalGapRound"] = f"{pooled.est:.0f}"
    values["FlagGapRound"] = f"{evidence.loc['ICLR 2024–27'].est:.0f}"
    values["JournalGapVarying"] = f"{int(pooled.varying):,}"
    meld = pd.read_csv(settings.REPORTS_DIR / "meld_summary.csv").set_index("name").value
    values["MeldThreshold"] = f"{meld.threshold:.2f}"
    for name in ["calibration_reviews", "intel_ai_reviews", "rewrite_reviews"]:
        values["Meld" + name.title().replace("_", "")] = f"{int(meld[name]):,}"
    for name in ["pre_source_max", "pre_year_max", "intel_ai_recall", "rewrite_recall_min"]:
        values["Meld" + name.title().replace("_", "")] = f"{meld[name]:.1f}"
    iclr_last = by_year[(by_year.conf == "ICLR") & (by_year.year == settings.ICLR_LAST)].iloc[0]
    values["IclrLastYear"] = str(settings.ICLR_LAST)
    record("IclrFlaggedLast", iclr_last.ai_flagged)
    icml = by_year[by_year.conf == "ICML"].set_index("year")
    record("IcmlFlaggedBase", icml.loc[2025].ai_flagged)
    record("IcmlFlaggedLast", icml.loc[2026].ai_flagged)
    flags = kept_meld_flags().flag
    pairs = analysis_pairs(model, None).copy()
    pairs["nflag"] = pairs.a.map(flags).astype(float) + pairs.b.map(flags).astype(float)
    main_conf, main_first, main_last = next(iter(settings.GAP_CELLS.values()))
    in_main = (pairs.conf == main_conf) & pairs.year.between(main_first, main_last)
    baseline = pairs[in_main & (pairs.nflag == 0)].groupby("panel_id").hit.mean() * 100
    record("FlagBaseline", baseline.mean(), baseline.sem(), len(baseline))
    placebo = pd.read_parquet(settings.ANALYSIS_DIR / f"placebo_outcomes_{model}.parquet")
    for year, prefix in [
        (2023, "CrossBase"),
        (2025, "CrossRecentBase"),
        (2026, "CrossRecentLast"),
        (settings.ICLR_LAST, "CrossLast"),
    ]:
        sample = (
            placebo[
                (placebo.conf == "ICLR") & (placebo.year == year) & (placebo.placebo_kind == SAME_VENUE_PLACEBO)
            ].hit_rate_strict.dropna()
            * 100
        )
        record(prefix, sample.mean(), sample.sem(), len(sample))
    embed = calibrate_embedding(pd.read_csv(reports / f"embed_{model}.csv")).set_index(["conf", "year"])
    for year, suffix in [(2020, "Base"), (2023, "Mid"), (settings.ICLR_LAST - 1, "Prev"), (settings.ICLR_LAST, "Last")]:
        row = embed.loc[("ICLR", year)]
        record("EmbedSame" + suffix, row.intra, row.intra_se, row.papers, digits=2)
        record("EmbedArea" + suffix, row.inter_area, digits=2)
        record("EmbedScaledSame" + suffix, row.calibrated_intra, row.calibrated_intra_se, row.papers, digits=2)
        record("EmbedScaledArea" + suffix, row.calibrated_inter_area, digits=2)
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
    index = ["kind", "conf", "years", "stratum", "split"]
    total = ("all", "all", "all", "all")
    ref = pd.read_csv(reports / f"extraction_{model}.csv").set_index(index)
    same, cross = ref.loc[("same", *total)], ref.loc[("cross", *total)]
    values["RefPanels"] = f"{int(same.panels)}"
    values["RefCrossPanels"] = f"{int(cross.panels)}"
    values["RefReviews"] = f"{int(same.reviews + cross.reviews)}"
    values["RefClaimsLuna"] = f"{same.claims_model:.1f}"
    values["RefClaimsCouncil"] = f"{same.claims_council:.1f}"
    for name, prefix in [
        ("f1_model", "RefSpanFOne"),
        ("f1_council", "RefCouncilSpanFOne"),
        ("precision_pooled", "RefPrecisionPooled"),
        ("recall_majority", "RefRecallMajority"),
        ("recall_unanimous", "RefRecallUnanimous"),
        ("aligned_model", "RefAligned"),
        ("aligned_council", "RefCouncilAligned"),
        ("category_model", "RefCategory"),
        ("category_council", "RefCouncilCategory"),
        ("target_model", "RefTarget"),
        ("target_council", "RefCouncilTarget"),
        ("specific_model", "RefSpecific"),
        ("specific_council", "RefCouncilSpecific"),
        ("claim_f1_model", "RefClaimFOne"),
        ("claim_f1_council", "RefCouncilClaimFOne"),
        ("cover_recall_model", "RefCoverRecall"),
        ("cover_precision_model", "RefCoverPrecision"),
        ("cover_council", "RefCouncilCover"),
    ]:
        for suffix in ("", "_low", "_high"):
            values[prefix + suffix.title()[1:]] = f"{same[name + suffix]:.1f}"
    council = pd.read_csv(reports / f"council_agreement_{model}.csv").set_index(index)
    pairs = council.loc[("same", *total)]
    values["RefPairPanels"] = f"{int(pairs.panels)}"
    for name in ["tp", "fp", "fn", "candidates"]:
        values["RefPair" + name.title()] = f"{int(pairs['pair_' + name]):,}"
    values["RefPairAllReviews"] = f"{int(pairs.reviews):,}"
    values["RefPairAllCandidates"] = f"{int(pairs.candidate_pairs):,}"
    values["RefPairCoverage"] = f"{100 * pairs.voted_pairs / pairs.candidate_pairs:.1f}"
    values["RefPairSplit"] = f"{int(pairs.split_pairs):,}"
    values["RefPairInvalid"] = f"{int(pairs.invalid_pairs):,}"
    values["RefPairLunaPositive"] = f"{int(pairs.retained_model_positives)}"
    values["RefPairAllLunaPositive"] = f"{int(pairs.model_positives)}"
    for name in ["precision", "recall", "f1"]:
        for suffix in ["", "_low", "_high"]:
            values["RefPair" + name.title().replace("1", "One") + suffix.title()[1:]] = f"{pairs['pair_' + name + suffix]:.1f}"
    for stratum in ["plain", "flagged"]:
        values["RefRatio" + stratum.title()] = f"{council.loc[('same', 'all', 'all', stratum, 'all')].pair_link_ratio:.0f}"
    council_types = pd.read_csv(reports / f"council_pair_types_{model}.csv").set_index("pair_type")
    for kind, name in [("HH", "None"), ("LL", "Two")]:
        values["RefTypeRatio" + name] = f"{council_types.loc[kind].pair_link_ratio:.0f}"
    values["RefTypeLinksNone"] = f"{int(council_types.loc['HH', ['pair_tp', 'pair_fn']].sum())}"
    swap = pd.read_csv(reports / f"grouper_swap_{model}.csv").set_index("grouper")
    others = swap.drop(index=model)
    values["RefSwapPanels"] = f"{int(swap.panels.iloc[0])}"
    for name, prefix in [("two", "RefSwapFlag"), ("panel_gap", "RefSwapPanel")]:
        values[prefix + "Luna"] = f"{swap.loc[model, name]:.1f}"
        values[prefix + "Min"] = f"{others[name].min():.1f}"
        values[prefix + "Max"] = f"{others[name].max():.1f}"
    pipes = pd.read_csv(reports / f"pipelines_{model}.csv").set_index(index)
    same_pipes, cross_pipes = pipes.loc[("same", *total)], pipes.loc[("cross", *total)]
    values["RefPipePanels"] = f"{int(same_pipes.panels)}"
    values["RefPairEmptyPanels"] = f"{int(same_pipes.panels - pairs.panels)}"
    values["RefPipeLuna"] = f"{same_pipes.overlap_model:.1f}"
    values["RefPipeCouncil"] = f"{same_pipes.overlap_council:.1f}"
    values["RefPipeCrossLuna"] = f"{cross_pipes.overlap_model:.1f}"
    values["RefPipeCrossCouncil"] = f"{cross_pipes.overlap_council:.1f}"
    for suffix in ("", "_low", "_high"):
        values["RefPipeDiff" + suffix.title()[1:]] = f"{same_pipes['diff_model' + suffix]:.1f}"
    values["RefPipeAbsDiffLuna"] = f"{same_pipes.absdiff_model:.1f}"
    values["RefPipeAbsDiffCouncil"] = f"{same_pipes.absdiff_council:.1f}"
    contrasts = pd.read_csv(reports / f"pipelines_contrasts_{model}.csv").set_index("pipeline")
    luna, members = contrasts.loc[model], contrasts.drop(index=model)
    values["RefPipeLevelMemberMin"] = f"{members.level.min():.1f}"
    values["RefPipeLevelMemberMax"] = f"{members.level.max():.1f}"
    for name, prefix, digits in [
        ("flag_gap", "RefPipeFlag", 1),
        ("iclr_slope", "RefPipeSlope", 2),
        ("iclr_step", "RefPipeStep", 1),
    ]:
        for suffix in ("", "_low", "_high"):
            values[prefix + "Luna" + suffix.title()[1:]] = f"{luna[name + suffix]:.{digits}f}"
        values[prefix + "MemberMin"] = f"{members[name].min():.{digits}f}"
        values[prefix + "MemberMax"] = f"{members[name].max():.{digits}f}"
        values[prefix + "Positive"] = f"{int((contrasts[name] > 0).sum())}"
        values[prefix + "Significant"] = f"{int((contrasts[name + '_low'] > 0).sum())}"
    pair_gap_low = contrasts.pair_gap - Z95 * contrasts.pair_gap_se
    values["RefPipePairGapLuna"] = f"{luna.pair_gap:.1f}"
    values["RefPipePairGapMemberMin"] = f"{members.pair_gap.min():.1f}"
    values["RefPipePairGapMemberMax"] = f"{members.pair_gap.max():.1f}"
    values["RefPipePairGapSignificant"] = f"{int((pair_gap_low > 0).sum())}"
    for name, prefix in [("kim", "Kim"), ("claimcheck", "Claimcheck")]:
        row = pd.read_csv(reports / f"{name}_{model}.csv").iloc[0]
        values[prefix + "Reviews"] = f"{int(row.reviews):,}"
        values[prefix + "Items"] = f"{int(row['items']):,}"
        for column, label in [("item_recall", "ItemRecall"), ("item_split", "ItemSplit"), ("claim_unmatched", "ClaimUnmatched")]:
            values[prefix + label] = f"{row[column]:.1f}"
            values[prefix + label + "Low"] = f"{row[column + '_low']:.1f}"
            values[prefix + label + "High"] = f"{row[column + '_high']:.1f}"
        values[prefix + "ClaimsBias"] = f"{row.claims_bias:.1f}"
    ident = pd.read_csv(reports / f"identification_{model}.csv").set_index("label")
    for label, prefix in [
        ("ICLR 2024–27", "IdIclr"),
        ("NeurIPS 2024–25", "IdNeurips"),
        ("ICML 2025–26", "IdIcml"),
        ("All journals 2024–26", "IdJournal"),
    ]:
        row = ident.loc[label]
        values[prefix + "Papers"] = f"{int(row.papers):,}"
        values[prefix + "Varying"] = f"{int(row.varying):,}"
        values[prefix + "TwoReports"] = f"{int(row.reports_2):,}"
        record(prefix + "Direct", row.endpoint, row.endpoint_se, row.endpoint_papers)
        values[prefix + "Fe"] = f"{row.two:.1f}"
        values[prefix + "DirectLow"] = f"{row.endpoint_low:.1f}"
        values[prefix + "DirectHigh"] = f"{row.endpoint_high:.1f}"
    recent = pd.read_csv(reports / f"robust_{model}.csv").query("base_year == @settings.CONTRAST_YEARS[0]")
    values["RobustMin"] = f"{recent.change.min():.1f}"
    values["RobustMax"] = f"{recent.change.max():.1f}"
    matched = pd.read_csv(reports / f"robust_{model}.csv").query("base_year == @settings.ICLR_SNAPSHOT_YEARS[0]")
    for check, prefix in [("Overlap", "IclrSnap"), (settings.WITHDRAWN_CHECK, "IclrKept")]:
        row = matched.set_index("check").loc[check]
        values[prefix + "Base"] = f"{row.base:.1f}"
        values[prefix + "Last"] = f"{row['last']:.1f}"
        record(prefix + "Change", row.change, row.se)
    left = pd.read_csv(reports / f"withdrawn_{model}.csv").set_index("name").value
    values["WithdrawnShare"] = f"{left.left_share:.0f}"
    values["WithdrawnEarlyShare"] = f"{left.last_left_share:.0f}"
    values["WithdrawnPapers"] = f"{int(left.left_papers):,}"
    record("WithdrawnLevel", left.left_level, left.left_level_se)
    values["RobustSnapMin"] = f"{matched.change.min():.1f}"
    values["RobustSnapMax"] = f"{matched.change.max():.1f}"
    sizes = pd.read_csv(reports / f"pair_type_sizes_{model}.csv").set_index("pair_type")
    for pair_type, row in sizes.iterrows():
        prefix = "Size" + pair_type.title()
        for name, label in [("shared", "Shared"), ("smaller", "Smaller"), ("union", "Union"), ("shared_adjusted", "SharedAdj")]:
            values[prefix + label] = f"{row[name]:.2f}" if pair_type == "HH" else f"{row[name]:+.2f}"
            if pair_type != "HH":
                values[prefix + label + "SE"] = f"{row[name + '_se']:.2f}"
                values[prefix + label + "Low"] = f"{row[name] - Z95 * row[name + '_se']:+.2f}"
                values[prefix + label + "High"] = f"{row[name] + Z95 * row[name + '_se']:+.2f}"
    types = pd.read_csv(reports / f"pair_types_{model}.csv").set_index(["conf", "pair_type"])
    for (conf, pair_type), row in types.iterrows():
        record(f"Type{pair_type.title()}{GAP_PREFIX[conf]}", row.est, row.se, row.papers)
    steps = pd.read_csv(reports / f"decomposition_{model}.csv").set_index(["conf", "year"])
    step, step_base = steps.loc[("ICLR", settings.CONTRAST_YEARS[1])], steps.loc[("ICLR", settings.CONTRAST_YEARS[0])]
    values["DecIclrMix"] = f"{step.composition:.1f}"
    values["DecIclrUnflagged"] = f"{step.within_unflagged:.1f}"
    values["DecIclrMixEarly"] = f"{step_base.composition:.1f}"
    step_snap = steps.loc[("ICLR", settings.ICLR_SNAPSHOT_YEARS[1])]
    values["DecSnapMix"] = f"{step_snap.composition:.1f}"
    values["DecSnapNoneLast"] = f"{step_snap.overlap_HH:.1f}"
    values["DecIclrNoneBase"] = f"{step_base.overlap_HH:.1f}"
    values["DecIclrNoneLast"] = f"{step.overlap_HH:.1f}"
    values["DecIclrCleanBase"] = f"{step_base.overlap_clean:.1f}"
    values["DecIclrCleanLast"] = f"{step.overlap_clean:.1f}"
    values["DecSnapCleanLast"] = f"{step_snap.overlap_clean:.1f}"
    values["DecIclrTwoShareBase"] = f"{step_base.share_LL:.1f}"
    subsets = pd.read_csv(reports / f"journal_subsets_{model}.csv").set_index("subset")
    for subset, prefix in [("Identifying", "JsIdent"), ("Two reports", "JsTwo")]:
        row = subsets.loc[subset]
        values[prefix + "Papers"] = f"{int(row.papers):,}"
        for name in ["chars", "claims", "overlap", "flagged", "share_F1000"]:
            record(prefix + name.title().replace("_", "").replace("1000", "Thousand"), row[name])
    cover = pd.read_csv(reports / f"coverage_{model}.csv").set_index("year")
    values["CovPapersMin"] = f"{int(cover.papers.min()):,}"
    values["CovPapersMax"] = f"{int(cover.papers.max()):,}"
    for year, suffix in [(2024, "First"), (2025, "Base"), (2026, "Last"), (settings.ICLR_SNAPSHOT_YEARS[1], "Snap")]:
        row = cover.loc[year]
        record("CovProblems" + suffix, row.problems, row.problems_se, row.papers)
        record("CovPerReview" + suffix, row.per_review, row.per_review_se)
        record("CovNew" + suffix, row.new_by_third, row.new_by_third_se, digits=2)
        record("CovShared" + suffix, row.shared_per_pair, row.shared_per_pair_se, digits=2)
    rewrite = pd.read_csv(reports / f"rewrite_test_{model}.csv").set_index(["condition", "group"])
    meld_rates = pd.read_csv(settings.REPORTS_DIR / "meld_rates.csv")
    for condition, base, meld_set in REWRITE_MACROS:
        for group, suffix in [("All", ""), ("ICLR", "Iclr"), ("Journals", "Journal")]:
            if (condition, group) not in rewrite.index:
                continue
            row = rewrite.loc[(condition, group)]
            prefix = base + suffix
            values[prefix + "Papers"] = f"{int(row.papers):,}"
            record(prefix + "TwoBefore", row.two_before)
            record(prefix + "TwoAfter", row.two_after)
            values[prefix + "TwoDiff"] = f"{row.two_diff:+.1f}"
            values[prefix + "TwoDiffLow"] = f"{row.two_diff_low:+.1f}"
            values[prefix + "TwoDiffHigh"] = f"{row.two_diff_high:+.1f}"
            record(prefix + "Two", row.two, row.two_se)
            record(prefix + "One", row.one, row.one_se)
            for name, label in [("unchanged_diff", "Unchanged"), ("alone_diff", "Alone")]:
                record(prefix + label, row[name], row[name + "_se"])
                values[prefix + label + "Low"] = f"{row[name + '_low']:.1f}"
                values[prefix + label + "High"] = f"{row[name + '_high']:.1f}"
        row = rewrite.loc[(condition, "All")]
        record(base + "Regroup", row.regroup_diff, row.regroup_diff_se)
        record(base + "LunaRecall", row.luna_recall)
        record(base + "ClaimsOriginal", row.claims_original)
        record(base + "ClaimsRewrite", row.claims_rewrite)
        texts = meld_rates[meld_rates.set == meld_set]
        record(base + "Flagged", (texts.flagged * texts.n).sum() / texts.n.sum())
    gap, effect = evidence.loc["ICLR 2024–27"], rewrite.loc[("outline_second", "ICLR")]
    record("RkFlagDiff", gap.est - effect.two, np.hypot(gap.se, effect.two_se))
    labels = pd.read_csv(settings.REPORTS_DIR / "meld_labels.csv").set_index("known")
    for known, prefix, digits in [
        ("polish", "Polish", 1),
        ("rewrite", "Rewrite", 1),
        ("intel_ai", "Intel", 1),
        ("heldout", "Heldout", 2),
        ("paper_review", "PaperLuna", 1),
        ("paper_review_second", "PaperKimi", 1),
    ]:
        row = labels.loc[known]
        values[prefix + "Texts"] = f"{int(row.texts):,}"
        record(prefix + "Flagged", 100 - row.unflagged, digits=digits)
        record(prefix + "Unflagged", row.unflagged, digits=digits)
    gigacheck = pd.read_csv(settings.REPORTS_DIR / "gigacheck_summary.csv").set_index("name").value
    values["GigacheckThreshold"] = f"{gigacheck.threshold:.2f}"
    values["GigacheckAgreementReviews"] = f"{int(gigacheck.meld_reviews):,}"
    values["GigacheckKappa"] = f"{gigacheck.meld_kappa:.2f}"
    for name, macro in [
        ("meld_both", "GigacheckBoth"),
        ("meld_only", "GigacheckMeldOnly"),
        ("gigacheck_not_meld", "GigacheckOnly"),
    ]:
        record(macro, gigacheck[name])
    gigacheck_labels = pd.read_csv(settings.REPORTS_DIR / "gigacheck_labels.csv").set_index("known").flagged
    record("GigacheckHeldoutFlagged", gigacheck_labels["heldout"], digits=2)
    second = pd.read_csv(reports / f"second_detector_{model}.csv").set_index("cell")
    iclr = second.loc["ICLR 2024–27"]
    record("GigacheckFlagGap", iclr.gigacheck, iclr.gigacheck_se, iclr.papers)
    repro = pd.read_csv(reports / f"repro_{model}.csv").set_index("name").value
    for name in ["pass1_empty", "pass1_error_rows", "pass2_error_rows", "calibration_in_analysis", "calibration_outside"]:
        values["Repro" + name.title().replace("_", "").replace("1", "One").replace("2", "Two")] = f"{int(repro[name]):,}"
    pd.DataFrame(rows).to_csv(reports / f"paper_numbers_{model}.csv", index=False)
    used = set(re.findall(r"\\([A-Za-z]+)", settings.PAPER_TEX.read_text()))
    Path(tables, "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items() if name in used) + "\n"
    )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def prompts(tex=settings.PROMPTS_TEX):
    blocks = [
        ("Pass 1 (per review)", extract.pass1_instructions("ml")),
        ("Pass 2 (per panel)", render("pass2", paper=extract.PASS2_PAPER_WORDING["ml"])),
    ]
    parts = [
        f"\\noindent\\textbf{{{title}.}}\n\\begin{{lstlisting}}\n{body.strip()}\n\\end{{lstlisting}}\n" for title, body in blocks
    ]
    Path(tex).write_text("\n".join(parts))


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
            "table": table,
            "numbers": numbers,
            "prompts": prompts,
        }
    )
