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

from outcomes import ICLR_FROM, excluded_review_ids, in_main_sample, in_window, main_sample_review_ids
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
REVIEW_FILES = ["reviews.parquet", "icml_sample.parquet", "other_reviews.parquet"]
COCODET = ["cocodet_main.jsonl", "cocodet_icml.jsonl", "cocodet_other.jsonl"]
SHORT = {
    "+ length and claim count": "+ length, claim count",
    "CoCoDet labels, ICLR": "CoCoDet labels",
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
FIELD_LABEL_PLACEMENT = {
    "BMC-Med": ("above_bar", (0, 2), "center", "bottom"),
    "SciPost": ("below_bar", (0, -2), "center", "top"),
    "Copernicus": ("at_point", (4, -3), "left", "top"),
    "PLOS": ("at_point", (-4, 0), "right", "center"),
}
DEFAULT_FIELD_LABEL_PLACEMENT = ("at_point", (4, 0), "left", "center")


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


def likeness(figs=settings.FIGURES_DIR):
    share = pd.read_csv(settings.REPORTS_DIR / "llm_share.csv").query("conf == 'ICLR' and year >= @ICLR_FROM")
    flags = pd.read_parquet(settings.DATA_DIR / "detector_scores.parquet", columns=["review_id", "llm_flag"])
    reviews = pd.read_parquet(
        settings.DATA_DIR / "reviews.parquet", columns=["review_id", "conf", "year", "withdrawn", "decision"]
    )
    reviews = reviews[in_main_sample(reviews)].merge(flags, on="review_id")
    flagged = reviews.query("conf == 'ICLR' and year >= @ICLR_FROM").groupby("year").llm_flag.mean() * 100
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.1, 1.8))
    fig.subplots_adjust(left=0.12, right=0.84, top=0.97, bottom=0.14)
    shade_chatgpt_period(ax)
    ax.plot(share.year, share.alpha * 100, "o-", ms=3, color=COLORS["ink"])
    ax.plot(flagged.index, flagged, "s--", ms=3, color=COLORS["flag"])
    end_label(ax, 2026, share.alpha.iloc[-1] * 100, "word-use\nindex", dy=-2)
    end_label(ax, 2026, flagged.iloc[-1], "flagged", dy=2)
    ax.set_xlim(2019.6, 2026.4)
    ax.set_ylim(0, 50)
    few_ticks(ax)
    year_ticks(ax)
    ax.set_ylabel("Share of ICLR reviews (%)")
    save(fig, figs, "figA_likeness.pdf", COLUMN_WIDTH)


def ai(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    flags = (
        pd.read_parquet(settings.DATA_DIR / "detector_scores.parquet", columns=["review_id", "llm_flag"])
        .set_index("review_id")
        .llm_flag.astype(int)
    )
    pairs = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    pairs["pair_type"] = (pairs.a.map(flags) + pairs.b.map(flags)).map({0: "HH", 1: "HL", 2: "LL"})
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.1, 1.9))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.14)
    iclr = pairs[(pairs.conf == "ICLR") & (pairs.year >= 2024)]
    styles = {
        "HH": ("neither flagged", COLORS["grey"], "o"),
        "HL": ("one flagged", COLORS["flag_light"], "s"),
        "LL": ("both flagged", COLORS["flag"], "D"),
    }
    for k, (pair_type, (_, color, marker)) in enumerate(styles.items()):
        per_year = iclr[iclr.pair_type == pair_type].groupby(["year", "panel_id"]).hit.mean().groupby("year")
        mean = per_year.mean() * 100
        ci = 100 * Z95 * per_year.std() / np.sqrt(per_year.count())
        ax.errorbar(
            mean.index + (k - 1) * 0.16,
            mean,
            yerr=ci,
            fmt=marker,
            ms=3.5,
            color=color,
            elinewidth=1.0,
            label=styles[pair_type][0],
        )
        end_label(
            ax,
            mean.index[-1] + (k - 1) * 0.16,
            mean.iloc[-1],
            styles[pair_type][0],
            dy={"HH": -3, "HL": 3, "LL": 0}[pair_type],
        )
    few_ticks(ax)
    ax.set_xlim(2023.6, 2027.4)
    ax.set_ylim(0, 20)
    year_ticks(ax, (2024, 2025, 2026))
    ax.set_ylabel("Critique overlap (%)")
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
            labels.append(SHORT.get(row.label, row.label).replace("-", "–"))
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
    conferences = {
        "ICLR": ("ICLR", COLORS["ICLR"], "o", 0),
        "NeurIPS": ("NeurIPS", COLORS["NeurIPS"], "s", 0),
        "ICML": ("ICML", COLORS["ICML"], "D", 0.12),
    }
    journals = {
        "F1000": ("F1000", COLORS["F1000"], "o", 0),
        "PLOS": ("PLOS", COLORS["PLOS"], "s", 0),
        "Copernicus": ("Copernicus", COLORS["grey"], "^", 0),
        "SciPost": ("SciPost", COLORS["grey"], "v", 0),
        "BMC-Med": ("BMC Medicine", COLORS["grey"], "D", 0),
    }
    for ax, styles, title in [(top, conferences, "a  ML conferences"), (bottom, journals, "b  Journals")]:
        shade_chatgpt_period(ax)
        overlap_panel(ax, by_year, styles, 2026.65, 2.4)
        ax.set_ylim(5, 21)
        ax.set_yticks([5, 10, 15, 20])
        ax.set_ylabel("Critique overlap (%)")
        ax.set_title(title, fontweight="bold")
        ax.annotate("LLM era", (CHATGPT_BAND[0], 21), xytext=(3, -2), textcoords="offset points", ha="left", va="top", fontsize=6.5, color=COLORS["grey"], style="italic")
        ax.tick_params(axis="x", labelsize=6)
    bottom.set_xlim(2018.6, CHATGPT_BAND[1])
    bottom.set_xticks(range(2019, 2027))
    top.tick_params(labelbottom=True)
    bottom.set_xlabel("Review year")
    save(fig, figs, "fig1_trends.pdf", COLUMN_WIDTH)


def cross(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    placebo = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"placebo_outcomes_{model}.parquet"))
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.1, 1.7))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.14)
    shade_chatgpt_period(ax)
    rate_by_conf = {}
    for conf in ["ICLR", "NeurIPS"]:
        rows = placebo[(placebo.conf == conf) & (placebo.placebo_kind == SAME_VENUE_PLACEBO)]
        rate_by_conf[conf] = mean_line(ax, rows, "hit_rate_strict", COLORS[conf])
    iclr, neurips = rate_by_conf["ICLR"], rate_by_conf["NeurIPS"]
    end_label(ax, 2025, neurips.loc[2025, "mean"], "NeurIPS", COLORS["NeurIPS"], dx=0, dy=-8, ha="center", va="top")
    end_label(ax, ICLR_FROM, iclr.loc[ICLR_FROM, "bar_top"], "ICLR", COLORS["ICLR"], dx=-2, dy=2, ha="left", va="bottom")
    ax.set_ylim(-0.3, 3.5)
    ax.set_xlim(2019.6, 2026.5)
    few_ticks(ax)
    year_ticks(ax)
    ax.set_ylabel("Overlap (%)")
    save(fig, figs, "figA_cross.pdf", COLUMN_WIDTH)


def category_shifts(model):
    claims = pd.read_parquet(
        settings.DATA_DIR / f"claims_table_{model}.parquet", columns=["review_id", "conf", "year", "category"]
    )
    main_sample_reviews = main_sample_review_ids(settings.DATA_DIR)
    claims = claims[(claims.conf == "ICLR") & claims.review_id.isin(main_sample_reviews) & (claims.category != "other")]
    share = pd.crosstab(claims.category, claims.year, normalize="columns") * 100
    shift = pd.DataFrame({"pre": share[list(range(ICLR_FROM, 2023))].mean(axis=1), "post": share[2026]})
    shift["change"] = shift.post - shift.pre
    return shift.sort_values("change")


def categories(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    shift = category_shifts(model)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.33, 2.35))
    y = np.arange(len(shift))
    ax.hlines(y, shift.pre, shift.post, color=COLORS["light"], lw=1.4, zorder=1)
    ax.plot(shift.post, y, "o", ms=3.5, color=COLORS["ink"], zorder=2)
    ax.plot(shift.pre, y, "o", ms=3.5, mfc="white", color=COLORS["grey"], zorder=3)
    change_x = 27.5
    for yi, change in zip(y, shift.change):
        ax.text(change_x, yi, f"{change:+.1f}".replace("-", "−"), ha="right", va="center", fontsize=7, color=COLORS["ink"])
    top = len(shift) - 1
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", color=COLORS["grey"]),
        plt.Line2D([], [], ls="", marker="o", ms=3.5, color=COLORS["ink"]),
    ]
    ax.legend(
        handles,
        [f"{ICLR_FROM}–2022 (mean of years)", "2026"],
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        handletextpad=0.3,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    ax.text(change_x, top + 0.9, "change", ha="right", va="center", fontsize=7, color=COLORS["grey"])
    few_ticks(ax, "x")
    ax.set_yticks(y, [CATEGORY_NAMES.get(c, c) for c in shift.index])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 28)
    ax.set_ylim(-0.7, top + 1.4)
    ax.set_xlabel("Share of ICLR critique claims (%)")
    save(fig, figs, "fig3_categories.pdf", COLUMN_WIDTH)


def journal_change(by_year, conf):
    rows = by_year[by_year.conf == conf]
    pre, last = rows[rows.year.between(2019, 2023)], rows[rows.year == rows.year.max()].iloc[0]
    return dict(
        overlap_change=last.match - pre.match.mean(),
        ai_change=last.ai_wordshare - pre.ai_wordshare.mean(),
        se=np.sqrt(last.match_se**2 + (pre.match_se**2).sum() / len(pre) ** 2),
        papers=last.papers + pre.papers.sum(),
    )


def fields(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    by_year_rows = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"by_year_{model}.csv")
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.1, 2.0))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.17)
    for conf, name in JOURNALS.items():
        change = journal_change(by_year_rows, conf)
        dx, dy, ci = change["ai_change"], change["overlap_change"], Z95 * change["se"]
        ax.errorbar(dx, dy, yerr=ci, fmt="o", ms=3.5, color=COLORS["ink"], elinewidth=1.0)
        anchor, offset, ha, va = FIELD_LABEL_PLACEMENT.get(conf, DEFAULT_FIELD_LABEL_PLACEMENT)
        label_y = {"above_bar": dy + ci, "below_bar": dy - ci, "at_point": dy}[anchor]
        ax.annotate(name, (dx, label_y), xytext=offset, textcoords="offset points", fontsize=7, ha=ha, va=va)
    ax.axhline(0, color=COLORS["light"], lw=0.6, zorder=0)
    ax.set_xlim(-4, 32)
    ax.set_ylim(-4, 8)
    few_ticks(ax, "x", n=4)
    few_ticks(ax, n=4)
    ax.set_xlabel("Change in AI-likeness index (points)")
    ax.set_ylabel("Change in overlap (points)")
    save(fig, figs, "fig4_fields.pdf", COLUMN_WIDTH)


def journal_grid(figs, name, ylabel, ylim, draw):
    fig, grid = plt.subplots(2, 3, figsize=(PAGE_WIDTH - 0.1, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.92, bottom=0.13, wspace=0.28, hspace=0.65)
    axes = grid.ravel()
    for ax, (conf, title) in zip(axes, JOURNALS.items(), strict=False):
        shade_chatgpt_period(ax)
        draw(ax, conf)
        ax.set_ylim(*ylim)
        ax.set_xlim(2018.4, 2026.6)
        few_ticks(ax, n=4)
        year_ticks(ax, (2019, 2022, 2026))
        ax.set_title(title)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(ylabel)
    axes[5].remove()
    save(fig, figs, name, PAGE_WIDTH)


def journals(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    by_year = pd.read_csv(settings.ANALYSIS_REPORTS_DIR / f"by_year_{model}.csv")

    def draw(ax, conf):
        overlap_line(ax, by_year, conf, COLORS["ink"])

    journal_grid(figs, "figA_journals.pdf", "Overlap (%)", (0, 30), draw)


def journals_cross(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    other = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"other_panel_outcomes_{model}.parquet"))

    def draw(ax, conf):
        mean_line(ax, other[(other.conf == conf) & other.placebo], "hit_rate_strict", COLORS["ink"])

    journal_grid(figs, "figA_journals_cross.pdf", "Overlap (%)", (-0.4, 5.5), draw)


def area(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    placebo = in_window(pd.read_parquet(settings.ANALYSIS_DIR / f"placebo_outcomes_{model}.parquet"))
    iclr = placebo[placebo.conf == "ICLR"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH - 0.1, 1.7))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.14)
    shade_chatgpt_period(ax)
    same_venue = mean_line(ax, iclr[iclr.placebo_kind == SAME_VENUE_PLACEBO], "hit_rate_strict", COLORS["ink"])
    same_area = mean_line(ax, iclr[iclr.placebo_kind == "same_area3"], "hit_rate_strict", COLORS["ink"], ls="--", mfc="white")
    end_label(ax, 2023, same_area.loc[2023, "bar_top"], "same research area", dx=0, dy=2, ha="center", va="bottom")
    end_label(ax, 2022, same_venue.loc[2022, "mean"], "same venue and year", dx=0, dy=-9, ha="center", va="top")
    ax.set_xlim(2019.6, 2026.4)
    ax.set_ylim(-0.3, 7.5)
    few_ticks(ax)
    year_ticks(ax)
    ax.set_ylabel("Overlap (%)")
    save(fig, figs, "figA_area.pdf", COLUMN_WIDTH)


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
    for draw in [overlap, ai, ai_checks, categories, fields, cross, area, journals, journals_cross, embedding]:
        draw(figs, model)
    likeness(figs)


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

    word_share = pd.concat(
        [
            pd.read_csv(settings.REPORTS_DIR / "llm_share.csv"),
            pd.read_csv(settings.REPORTS_DIR / "llm_share_all.csv").query("conf not in ['ICLR', 'NeurIPS']"),
        ]
    )
    rows = rows.join(word_share.set_index(["conf", "year"]).alpha.mul(100).rename("ai_wordshare"))

    review_meta = pd.concat([pd.read_parquet(settings.DATA_DIR / f, columns=["review_id", "conf", "year"]) for f in REVIEW_FILES])
    review_meta = review_meta.drop_duplicates("review_id")
    review_meta = review_meta[~review_meta.review_id.isin(excluded_review_ids(settings.DATA_DIR))]
    cocodet = pd.concat([pd.read_json(settings.DATA_DIR / f, lines=True) for f in COCODET]).drop_duplicates("review_id")
    cocodet = cocodet.merge(review_meta, on="review_id")
    cocodet["flag"] = cocodet[["human", "mix", "ai"]].idxmax(axis=1) != "human"
    rows = rows.join(cocodet.groupby(["conf", "year"]).flag.mean().mul(100).rename("ai_cocodet"))
    style_flags = pd.read_parquet(settings.DATA_DIR / "detector_scores.parquet", columns=["review_id", "llm_flag"])
    style_flags = style_flags.merge(review_meta, on="review_id")
    rows = rows.join(style_flags.groupby(["conf", "year"]).llm_flag.mean().mul(100).rename("ai_flagged"))

    rows = in_window(rows.reset_index())
    rows["conf"] = pd.Categorical(rows.conf, ORDER)
    return rows.sort_values(["conf", "year"])


def tex_number(v):
    return "--" if pd.isna(v) else f"{v:.1f}"


def latex(rows):
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Source & Year & Papers & Reviews & Claims/rev. & Overlap (\%) & Other papers (\%) & AI-likeness (\%) & Flagged (\%) & CoCoDet (\%) \\",
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
                tex_number(row.ai_wordshare),
                tex_number(row.ai_flagged),
                tex_number(row.ai_cocodet),
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
        ("CoCoDet labels, ICLR", "CocoGap"),
    ]:
        row = evidence.loc[label]
        record(prefix, row.est, row.se, row.papers)
    flags = (
        pd.read_parquet(settings.DATA_DIR / "detector_scores.parquet", columns=["review_id", "llm_flag"])
        .set_index("review_id")
        .llm_flag
    )
    pairs = pd.read_parquet(settings.ANALYSIS_DIR / f"pairs_{model}.parquet")
    pairs["nflag"] = pairs.a.map(flags).astype(float) + pairs.b.map(flags).astype(float)
    baseline = pairs[(pairs.conf == "ICLR") & (pairs.year >= 2024) & (pairs.nflag == 0)].groupby("panel_id").hit.mean() * 100
    record("FlagBaseline", baseline.mean(), baseline.sem(), len(baseline))
    record("FlagRelative", evidence.loc["ICLR 2024–26"].est / baseline.mean() * 100, digits=0)
    placebo = pd.read_parquet(settings.ANALYSIS_DIR / f"placebo_outcomes_{model}.parquet")
    for year, prefix in [(2023, "CrossBase"), (2026, "CrossLast")]:
        sample = (
            placebo[
                (placebo.conf == "ICLR") & (placebo.year == year) & (placebo.placebo_kind == SAME_VENUE_PLACEBO)
            ].hit_rate_strict.dropna()
            * 100
        )
        record(prefix, sample.mean(), sample.sem(), len(sample))
    categories = category_shifts(model)
    categories.to_csv(reports / f"category_shift_{model}.csv")
    for category, prefix in [
        ("clarity/presentation", "Clarity"),
        ("typo/formatting", "Typos"),
        ("experiments-scope", "Experiments"),
    ]:
        row = categories.loc[category]
        record(prefix + "Base", row.pre, digits=0)
        record(prefix + "Last", row.post, digits=0)
    shifts = []
    for conf, prefix in [
        ("F1000", "FResearch"),
        ("PLOS", "Plos"),
        ("SciPost", "Scipost"),
        ("BMC-Med", "Bmc"),
        ("Copernicus", "Copernicus"),
    ]:
        change = journal_change(by_year, conf)
        record(prefix + "Change", change["overlap_change"], change["se"], change["papers"])
        shifts.append(dict(source=conf, overlap_change=change["overlap_change"], ai_change=change["ai_change"], se=change["se"]))
    shifts = pd.DataFrame(shifts)
    shifts.to_csv(reports / f"journal_changes_{model}.csv", index=False)
    correlation = shifts.overlap_change.corr(shifts.ai_change)
    record("JournalCorrelation", correlation, n=len(shifts), digits=2)
    limits = np.tanh(np.arctanh(correlation) + np.array([-1, 1]) * Z95 / np.sqrt(len(shifts) - 3))
    values["JournalCorrelationLow"], values["JournalCorrelationHigh"] = [f"{x:.2f}" for x in limits]
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
            "area": area,
            "likeness": likeness,
            "cross": cross,
            "journals": journals,
            "journals_cross": journals_cross,
            "table": table,
            "numbers": numbers,
        }
    )
