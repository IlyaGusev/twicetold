"""Paper figures (paper/figs/*.pdf) and the per-source-year table (reports/by_year_<model>.csv, paper/tables/by_year_*.tex).

Style follows paper/FIGURE_GUIDE.md; the color map is in paper/PAPER_NOTES.md. Figures are drawn at their final width
(COL or PAGE inches) and included in LaTeX without scaling. ICLR years before outcomes.ICLR_FROM are left out.

Inputs: reports/match_std_<model>.csv (outcomes.py standardize), reports/rq1_evidence_<model>.csv (validate.py rq1_evidence),
reports/by_year_<model>.csv (run `table` first), reports/embed_<model>.csv, reports/llm_share.csv, data/ pairs, placebo panels,
claims table and detector scores. Every model-specific input follows --model; regenerate them with the same model first.

Usage:
  uv run python -I src/figures.py all [--figs DIR] [--model M]          # every figure below
  uv run python -I src/figures.py ai|overlap|categories|fields [--figs DIR]   # main figures
  uv run python -I src/figures.py likeness|cross|area|journals|journals_cross|embedding   # appendix figures
  uv run python -I src/figures.py table [--csv reports/by_year_<model>.csv --tables paper/tables --model M]
"""

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

from outcomes import ICLR_FROM, in_window
from settings import settings

COL, PAGE = 3.03, 6.3
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "errorbar.capsize": 0, "pdf.fonttype": 42,
})
# Okabe-Ito; meanings fixed across the paper (PAPER_NOTES.md)
C = {"ICLR": "#0072B2", "NeurIPS": "#D55E00", "ICML": "#009E73", "flag": "#E69F00", "flag_light": "#F3CF7F",
     "grey": "#8C8C8C", "light": "#C8C8C8", "ink": "#222222", "band": "#EFEFEF"}
ORDER = ["ICLR", "NeurIPS", "ICML", "SciPost", "F1000", "PLOS", "BMC-Med", "Copernicus"]
JOURNALS = {"SciPost": "SciPost", "F1000": "F1000", "PLOS": "PLOS", "BMC-Med": "BMC Medicine", "Copernicus": "Copernicus"}
# the main CoCoDet runs; other cocodet_*.jsonl files are truncation and style-swap checks
COCODET = ["cocodet_main.jsonl", "cocodet_icml.jsonl", "cocodet_other.jsonl"]
SHORT = {"+ length and claim count": "+ length, claim count", "CoCoDet labels, ICLR": "CoCoDet labels",
         "Main model, 500 ICLR 2026 papers": "Main model, 500 papers", "Second model, same papers": "Second model, same 500",
         "End to end, 50 papers": "End to end (50 papers)", "After style swap, 48 papers": "Style swap (48)",
         "Pair audit, 60 papers": "Pair audit (60)", "Top-45% scores, ICLR 2020": "ICLR 2020",
         "Top-45% scores, ICLR 2021-2022": "ICLR 2021–22", "Top-45% scores, ICLR 2023": "ICLR 2023"}
CATEGORY_NAMES = {"clarity/presentation": "Clarity, presentation", "experiments-scope": "Scope of experiments",
                  "baselines/comparisons": "Baselines", "theory/correctness": "Theory, correctness",
                  "reproducibility/details": "Missing details", "typo/formatting": "Typos, formatting",
                  "novelty": "Novelty", "related-work": "Related work", "assumptions": "Assumptions",
                  "unsupported-claim": "Unsupported claim", "ablation": "Ablations", "weak-results": "Weak results",
                  "compute-cost/scalability": "Compute cost", "generalization": "Generalization",
                  "statistical-rigor": "Statistical rigor", "limitations/societal": "Limitations, societal"}


def chatgpt(ax, right=2026.5):
    """The same light band in every time plot: review cycles from 2023 on (ICLR 2023 reviews predate ChatGPT)."""
    ax.axvspan(2022.6, right, color=C["band"], zorder=0, lw=0)


def years(ax, ticks=(2020, 2023, 2026)):
    ax.set_xticks(list(ticks), [f"'{t % 100:02d}" for t in ticks])


def line(ax, df, col, color, scale=100, **kw):
    g = df.groupby("year")[col]
    m, e = g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(g.count())
    ax.errorbar(m.index, m * scale, yerr=e * scale, fmt="o", ms=3, color=color, elinewidth=1.0, **{"ls": "-", **kw})
    return m * scale


def std_line(ax, S, conf, color, dx=0.0, **kw):
    s = S[S.conf == conf]
    ax.errorbar(s.year + dx, s["std"], yerr=1.96 * s.std_se, fmt="o-", ms=3, color=color, elinewidth=1.0, **kw)
    return s.set_index("year")["std"]


def end_label(ax, x, y, text, color=C["ink"], dx=3, dy=0, ha="left", va="center"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=color, fontsize=7)


def few_ticks(ax, axis="y", n=5):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_locator(MaxNLocator(n, integer=True))


def save(fig, figs, name, width):
    """Crop to the drawn content, then pad to exactly `width` inches so LaTeX includes the figure unscaled."""
    # axis limits are fixed by hand, so warn when another model's numbers fall outside them
    for ax in fig.axes:
        d, (x0, x1), (y0, y1) = ax.dataLim, sorted(ax.get_xlim()), sorted(ax.get_ylim())
        # the ChatGPT band (the only patch) runs past the x limits by design
        if d.y0 < y0 or d.y1 > y1 or not ax.patches and (d.x0 < x0 or d.x1 > x1):
            print(f"WARNING {name}: data x {d.x0:.2f}..{d.x1:.2f}, y {d.y0:.2f}..{d.y1:.2f} outside the axis limits")
    figs = Path(figs)
    figs.mkdir(parents=True, exist_ok=True)
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.02)
    if bb.width > width + 1e-3:
        print(f"WARNING {name}: content {bb.width:.2f} in wider than {width:.2f} in; shrink figsize")
    extra = max(0.0, width - bb.width) / 2
    fig.savefig(figs / name, bbox_inches=Bbox([[bb.x0 - extra, bb.y0], [bb.x1 + extra, bb.y1]]))
    plt.close(fig)


def likeness(figs=settings.FIGURES_DIR):
    """Appendix figure: ICLR AI-likeness index and share of flagged reviews by year."""
    A = pd.read_csv(settings.REPORTS_DIR / "llm_share.csv").query("conf == 'ICLR' and year >= @ICLR_FROM")
    D = pd.read_parquet(settings.DATA_DIR / "detector_scores.parquet", columns=["review_id", "llm_flag"])
    R = pd.read_parquet(settings.DATA_DIR / "reviews.parquet", columns=["review_id", "conf", "year"]).merge(D, on="review_id")
    flag = R.query("conf == 'ICLR' and year >= @ICLR_FROM").groupby("year").llm_flag.mean() * 100
    fig, ax = plt.subplots(figsize=(COL - 0.1, 1.8))
    fig.subplots_adjust(left=0.12, right=0.84, top=0.97, bottom=0.14)
    chatgpt(ax)
    ax.plot(A.year, A.alpha * 100, "o-", ms=3, color=C["ink"])
    ax.plot(flag.index, flag, "s--", ms=3, color=C["flag"])
    end_label(ax, 2026, A.alpha.iloc[-1] * 100, "word-use\nindex", dy=-2)
    end_label(ax, 2026, flag.iloc[-1], "flagged", dy=2)
    ax.set_xlim(2019.6, 2026.4)
    ax.set_ylim(0, 50)
    few_ticks(ax)
    years(ax)
    ax.set_ylabel("Share of ICLR reviews (%)")
    save(fig, figs, "figA_likeness.pdf", COL)


def ai(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Fig. 1: match rate by pair type, and every estimate of the within-paper gap."""
    flags = pd.read_parquet(settings.DATA_DIR / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag.astype(int)
    P = pd.read_parquet(settings.DATA_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    P["pair_type"] = (P.a.map(flags) + P.b.map(flags)).map({0: "HH", 1: "HL", 2: "LL"})
    E = pd.read_csv(settings.REPORTS_DIR / f"rq1_evidence_{model}.csv")

    fig = plt.figure(figsize=(PAGE - 0.15, 2.15))
    # manual positions: (a) carries labels right of its axis, (b) long row labels on its left
    axes = [fig.add_axes(r) for r in ([0.06, 0.14, 0.3, 0.74], [0.66, 0.14, 0.33, 0.74])]
    ax = axes[0]
    p = P[(P.conf == "ICLR") & (P.year >= 2024)]
    styles = {"HH": ("neither", C["grey"], "o"), "HL": ("one", C["flag_light"], "s"), "LL": ("both", C["flag"], "D")}
    last = {}
    for k, (t, (lab, color, marker)) in enumerate(styles.items()):
        # one value per paper and type first, so the CI reflects papers, not correlated pairs within a paper
        g = p[p.pair_type == t].groupby(["year", "panel_id"]).hit.mean().groupby("year")
        m, e = g.mean() * 100, 196 * g.std() / np.sqrt(g.count())
        ax.errorbar(m.index + (k - 1) * 0.16, m, yerr=e, fmt=marker, ms=3.5, color=color, elinewidth=1.0)
        last[t] = m.iloc[-1]
    # "one" and "neither" sit close on a zero-based axis, so their labels are pushed apart
    lo, hi = sorted(["HH", "HL"], key=last.get)
    for t, dy in {lo: -0.5, hi: 0.5, "LL": 0}.items():
        ax.text(2026.5, last[t] + dy, styles[t][0], fontsize=7, va="center", color=C["ink"])
    ax.text(2026.5, 16.3, "flagged:", fontsize=7, va="center", color=C["grey"])
    few_ticks(ax)
    ax.set_xlim(2023.6, 2027.6)
    ax.set_ylim(0, 17)
    ax.set_xticks([2024, 2025, 2026], ["'24", "'25", "'26"])
    ax.set_title("(a) Match rate by pair, ICLR (%)")

    ax = axes[1]
    y, ticks, labels, bold = 0, [], [], []
    for gname, g in E.groupby("group", sort=False):
        ticks.append(y)
        labels.append(gname)
        bold.append(True)
        y -= 1
        placebo = gname.startswith("Placebo")
        color = C["grey"] if placebo else C["ink"]
        for r in g.itertuples():
            ax.errorbar(r.est, y, xerr=1.96 * r.se, fmt="s" if gname.startswith("Blind") else "o", ms=3.5,
                        color=color, mfc="white" if placebo else color, elinewidth=1.0)
            ticks.append(y)
            labels.append(SHORT.get(r.label, r.label).replace("-", "–"))
            bold.append(False)
            y -= 1
        y -= 0.4
    ax.axvline(0, color=C["light"], lw=0.6, zorder=0)
    ax.axvline(E.est.iloc[0], color=C["grey"], lw=0.6, ls=":", zorder=0)
    ax.set_yticks(ticks, labels)
    for t, b in zip(ax.get_yticklabels(), bold):
        t.set_fontweight("bold" if b else "normal")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlim(-2.5, 15)
    ax.set_xlabel("Match rate, both minus neither flagged")
    ax.set_title("(b) Within-paper gap (points)")
    save(fig, figs, "fig1_ai.pdf", PAGE)


def overlap(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Fig. 2: standardized same-paper match rate by venue."""
    S = pd.read_csv(settings.REPORTS_DIR / f"match_std_{model}.csv")
    fig, ax = plt.subplots(figsize=(COL - 0.1, 1.9))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13)
    chatgpt(ax)
    s = {conf: std_line(ax, S, conf, C[conf], dx) for conf, dx in [("ICLR", 0), ("NeurIPS", 0), ("ICML", 0.25)]}
    end_label(ax, ICLR_FROM, s["ICLR"].iloc[0], "ICLR", C["ICLR"], dx=-2, dy=13, ha="left")
    end_label(ax, 2022, s["NeurIPS"].loc[2022], "NeurIPS", C["NeurIPS"], dx=0, dy=-11, ha="center")
    end_label(ax, 2025.25, s["ICML"].iloc[0], "ICML", C["ICML"], dx=4, dy=-4, ha="left")
    ax.set_ylim(0, 13)
    ax.set_xlim(2019.6, 2026.5)
    few_ticks(ax)
    years(ax)
    ax.set_ylabel("Standardized match rate (%)")
    save(fig, figs, "fig2_overlap.pdf", COL)


def cross(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Appendix figure: match rate between reviews of different papers of one venue and year."""
    Q = in_window(pd.read_parquet(settings.DATA_DIR / f"placebo_outcomes_{model}.parquet"))
    fig, ax = plt.subplots(figsize=(COL - 0.1, 1.7))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.14)
    chatgpt(ax)
    q = {conf: line(ax, Q[(Q.conf == conf) & (Q.placebo_kind == "shuffled4")], "hit_rate_strict", C[conf])
         for conf in ["ICLR", "NeurIPS"]}
    end_label(ax, 2025, q["NeurIPS"].loc[2025], "NeurIPS", C["NeurIPS"], dx=0, dy=-8, ha="center", va="top")
    end_label(ax, ICLR_FROM, q["ICLR"].loc[ICLR_FROM], "ICLR", C["ICLR"], dx=-2, dy=8, ha="left", va="bottom")
    ax.set_ylim(-0.3, 1.8)
    ax.set_xlim(2019.6, 2026.5)
    few_ticks(ax)
    years(ax)
    ax.set_ylabel("Match rate (%)")
    save(fig, figs, "figA_cross.pdf", COL)


def categories(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Fig. 3: share of ICLR critique claims by category, 2020-2022 against 2026."""
    T = pd.read_parquet(settings.DATA_DIR / f"claims_table_{model}.parquet", columns=["review_id", "conf", "year", "category"])
    keep = set(pd.read_parquet(settings.DATA_DIR / "reviews.parquet", columns=["review_id"]).review_id)
    T = T[(T.conf == "ICLR") & T.review_id.isin(keep) & (T.category != "other")]
    f = pd.crosstab(T.category, T.year, normalize="columns") * 100
    d = pd.DataFrame({"pre": f[list(range(ICLR_FROM, 2023))].mean(axis=1), "post": f[2026]})
    d["change"] = d.post - d.pre
    d = d.sort_values("change")
    fig, ax = plt.subplots(figsize=(COL - 0.33, 2.35))
    y = np.arange(len(d))
    ax.hlines(y, d.pre, d.post, color=C["light"], lw=1.4, zorder=1)
    ax.plot(d.post, y, "o", ms=3.5, color=C["ink"], zorder=2)
    ax.plot(d.pre, y, "o", ms=3.5, mfc="white", color=C["grey"], zorder=3)
    for yi, change in zip(y, d.change):
        ax.text(27.5, yi, f"{change:+.1f}".replace("-", "−"), ha="right", va="center", fontsize=7, color=C["ink"])
    top = len(d) - 1
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", color=C["grey"]),
               plt.Line2D([], [], ls="", marker="o", ms=3.5, color=C["ink"])]
    ax.legend(handles, [f"{ICLR_FROM}–2022 (mean of years)", "2026"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0, frameon=False)
    ax.text(27.5, top + 0.9, "change", ha="right", va="center", fontsize=7, color=C["grey"])
    few_ticks(ax, "x")
    ax.set_yticks(y, [CATEGORY_NAMES.get(c, c) for c in d.index])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 28)
    ax.set_ylim(-0.7, top + 1.4)
    ax.set_xlabel("Share of ICLR critique claims (%)")
    save(fig, figs, "fig3_categories.pdf", COL)


def fields(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Fig. 4: change of each journal source's standardized match rate against the change in its AI-likeness index."""
    T = pd.read_csv(settings.REPORTS_DIR / f"by_year_{model}.csv")
    fig, ax = plt.subplots(figsize=(COL - 0.1, 2.0))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.17)
    # journal sources only: their AI-likeness comes from one field-matched reference, the ML venues' from another
    for conf, name in JOURNALS.items():
        g = T[T.conf == conf]
        pre, last = g[g.year.between(2019, 2023)], g[g.year == g.year.max()].iloc[0]
        dx = last.ai_wordshare - pre.ai_wordshare.mean()
        dy = last.match_std - pre.match_std.mean()
        se = np.sqrt(last.match_std_se ** 2 + (pre.match_std_se ** 2).sum() / len(pre) ** 2)
        ax.errorbar(dx, dy, yerr=1.96 * se, fmt="o", ms=3.5, color=C["ink"], elinewidth=1.0)
        # BMC Medicine and SciPost sit close together, so their labels go above and below the error bars
        y, off, ha, va = {"BMC-Med": (dy + 1.96 * se, (0, 2), "center", "bottom"), "SciPost": (dy - 1.96 * se, (0, -2), "center", "top"),
                          "Copernicus": (dy, (4, -3), "left", "top")}.get(conf, (dy, (4, 0), "left", "center"))
        ax.annotate(name, (dx, y), xytext=off, textcoords="offset points", fontsize=7, ha=ha, va=va)
    ax.axhline(0, color=C["light"], lw=0.6, zorder=0)
    ax.set_xlim(-4, 32)
    ax.set_ylim(-3, 9)
    few_ticks(ax, n=4)
    ax.set_xlabel("Change in AI-likeness index (points)")
    ax.set_ylabel("Change in std. match rate (points)")
    save(fig, figs, "fig4_fields.pdf", COL)


def journal_grid(figs, name, ylabel, ylim, draw):
    """One panel per journal source; draw(ax, conf) plots the series."""
    fig, grid = plt.subplots(2, 3, figsize=(PAGE - 0.1, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.92, bottom=0.13, wspace=0.28, hspace=0.65)
    axes = grid.ravel()
    for ax, (conf, title) in zip(axes, JOURNALS.items(), strict=False):
        chatgpt(ax)
        draw(ax, conf)
        ax.set_ylim(*ylim)
        ax.set_xlim(2018.4, 2026.6)
        few_ticks(ax, n=4)
        years(ax, (2019, 2022, 2026))
        ax.set_title(title)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(ylabel)
    axes[5].remove()
    save(fig, figs, name, PAGE)


def journals(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Appendix figure: standardized same-paper match rate per journal source and year."""
    S = pd.read_csv(settings.REPORTS_DIR / f"match_std_{model}.csv")
    journal_grid(figs, "figA_journals.pdf", "Std. match rate (%)", (0, 19), lambda ax, conf: std_line(ax, S, conf, C["ink"]))


def journals_cross(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Appendix figure: match rate between reports on different papers of one journal source and year."""
    other = in_window(pd.read_parquet(settings.DATA_DIR / f"other_panel_outcomes_{model}.parquet"))
    journal_grid(figs, "figA_journals_cross.pdf", "Match rate (%)", (-0.2, 3.4),
                 lambda ax, conf: line(ax, other[(other.conf == conf) & other.placebo], "hit_rate_strict", C["ink"]))


def area(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Appendix figure: ICLR match rate between reviews of different papers, same venue-year against same research area."""
    Q = in_window(pd.read_parquet(settings.DATA_DIR / f"placebo_outcomes_{model}.parquet"))
    Q = Q[Q.conf == "ICLR"]
    fig, ax = plt.subplots(figsize=(COL - 0.1, 1.7))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.14)
    chatgpt(ax)
    v = line(ax, Q[Q.placebo_kind == "shuffled4"], "hit_rate_strict", C["ink"])
    a = line(ax, Q[Q.placebo_kind == "same_area3"], "hit_rate_strict", C["ink"], ls="--", mfc="white")
    end_label(ax, 2023, a.loc[2023], "same research area", dx=0, dy=10, ha="center", va="bottom")
    end_label(ax, 2022, v.loc[2022], "same venue and year", dx=0, dy=-9, ha="center", va="top")
    ax.set_xlim(2019.6, 2026.4)
    ax.set_ylim(-0.3, 3.4)
    few_ticks(ax)
    years(ax)
    ax.set_ylabel("Match rate (%)")
    save(fig, figs, "figA_area.pdf", COL)


def embedding(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    """Appendix figure: cosine similarity of whole-review embeddings, same paper and different papers of one area."""
    E = in_window(pd.read_csv(settings.REPORTS_DIR / f"embed_{model}.csv"))
    fig, ax = plt.subplots(figsize=(COL - 0.5, 1.8))
    chatgpt(ax)
    for conf in ["ICLR", "NeurIPS"]:
        e = E[E.conf == conf]
        ax.plot(e.year, e.intra, "o-", ms=3, color=C[conf])
        ax.plot(e.year, e.inter_area, "o--", ms=3, mfc="white", color=C[conf])
    i, n = E[E.conf == "ICLR"], E[E.conf == "NeurIPS"]
    end_label(ax, 2026, i.intra.iloc[-1], "ICLR, same paper", C["ICLR"])
    end_label(ax, 2025, n.intra.iloc[-1], "NeurIPS, same paper", C["NeurIPS"], dy=6, ha="right", dx=0)
    end_label(ax, 2026, i.inter_area.iloc[-1], "ICLR, different\npapers, same area", C["ICLR"])
    end_label(ax, 2023, n.set_index("year").inter_area.loc[2023], "NeurIPS, different papers", C["NeurIPS"], dx=0, dy=-10, ha="center")
    ax.set_xlim(2019.6, 2026.4)
    ax.set_ylim(0, 0.9)
    years(ax)
    ax.set_ylabel("Cosine similarity")
    save(fig, figs, "figA_embedding.pdf", COL)


def all_figures(figs=settings.FIGURES_DIR, model=settings.DEFAULT_MODEL):
    for f in [ai, overlap, categories, fields, cross, area, journals, journals_cross, embedding]:
        f(figs, model)
    likeness(figs)


def summarize(df, placebo):
    g = df.groupby(["conf", "year"])
    out = pd.DataFrame({"papers": g.size(), "reviews": g.n_reviews.sum(), "claims_per_review": g.claims_per_review.mean(),
                        "match": g.hit_rate_strict.mean() * 100, "match_se": g.hit_rate_strict.sem() * 100})
    return out.join(placebo.groupby(["conf", "year"]).hit_rate_strict.mean().mul(100).rename("match_other_papers"))


def by_year(model=settings.DEFAULT_MODEL):
    """One row per source-year: sample size, claims per review, match rates and AI-text measures."""
    ml = pd.read_parquet(settings.DATA_DIR / f"panel_outcomes_{model}.parquet")
    ml_placebo = pd.read_parquet(settings.DATA_DIR / f"placebo_outcomes_{model}.parquet").query("placebo_kind == 'shuffled4'")
    icml = pd.read_parquet(settings.DATA_DIR / f"icml_panel_outcomes_{model}.parquet")
    other = pd.read_parquet(settings.DATA_DIR / f"other_panel_outcomes_{model}.parquet")
    T = pd.concat([summarize(ml, ml_placebo), summarize(icml, icml[icml.placebo]),
                   summarize(other[~other.placebo], other[other.placebo])])

    alpha = pd.concat([pd.read_csv(settings.REPORTS_DIR / "llm_share.csv"),
                       pd.read_csv(settings.REPORTS_DIR / "llm_share_all.csv").query("conf not in ['ICLR', 'NeurIPS']")])
    T = T.join(alpha.set_index(["conf", "year"]).alpha.mul(100).rename("ai_wordshare"))

    meta = pd.concat([pd.read_parquet(settings.DATA_DIR / f, columns=["review_id", "conf", "year"])
                      for f in ["reviews.parquet", "icml_sample.parquet", "other_reviews.parquet"]]).drop_duplicates("review_id")
    det = pd.concat([pd.read_json(settings.DATA_DIR / f, lines=True) for f in COCODET]).drop_duplicates("review_id").merge(meta, on="review_id")
    det["flag"] = det[["human", "mix", "ai"]].idxmax(axis=1) != "human"
    T = T.join(det.groupby(["conf", "year"]).flag.mean().mul(100).rename("ai_cocodet"))
    style = pd.read_parquet(settings.DATA_DIR / "detector_scores.parquet", columns=["review_id", "llm_flag"]).merge(meta, on="review_id")
    T = T.join(style.groupby(["conf", "year"]).llm_flag.mean().mul(100).rename("ai_flagged"))
    S = pd.read_csv(settings.REPORTS_DIR / f"match_std_{model}.csv").set_index(["conf", "year"])
    T = T.join(S[["std", "std_se"]].rename(columns={"std": "match_std", "std_se": "match_std_se"}))

    T = in_window(T.reset_index())
    T["conf"] = pd.Categorical(T.conf, ORDER)
    return T.sort_values(["conf", "year"])


def fmt(v, d=1):
    return "--" if pd.isna(v) else f"{v:.{d}f}"


def latex(T):
    lines = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r"Source & Year & Papers & Reviews & Claims/rev. & Match (\%) & Std. match (\%) & Other papers (\%) & AI-likeness (\%) & Flagged (\%) & CoCoDet (\%) \\",
             r"\midrule"]
    for conf, g in T.groupby("conf", observed=True):
        for k, r in enumerate(g.itertuples()):
            lines.append(f"{conf if k == 0 else ''} & {r.year} & {r.papers:,} & {r.reviews:,} & {fmt(r.claims_per_review)} & "
                         f"{fmt(r.match)} $\\pm$ {fmt(r.match_se)} & {fmt(r.match_std)} $\\pm$ {fmt(r.match_std_se)} & {fmt(r.match_other_papers)} & {fmt(r.ai_wordshare)} & "
                         f"{fmt(r.ai_flagged)} & {fmt(r.ai_cocodet)} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def table(csv=None, tables=settings.TABLES_DIR, model=settings.DEFAULT_MODEL):
    """reports/by_year_<model>.csv plus two LaTeX tables: by_year_ml.tex (conferences) and by_year_journals.tex."""
    T = by_year(model)
    T.to_csv(csv or settings.REPORTS_DIR / f"by_year_{model}.csv", index=False)
    print(T.round(2).to_string(index=False))
    ml = T.conf.isin(["ICLR", "NeurIPS", "ICML"])
    Path(tables, "by_year_ml.tex").write_text(latex(T[ml]))
    Path(tables, "by_year_journals.tex").write_text(latex(T[~ml]))


if __name__ == "__main__":
    fire.Fire({"all": all_figures, "ai": ai, "overlap": overlap, "categories": categories, "fields": fields,
               "embedding": embedding, "area": area, "likeness": likeness, "cross": cross, "journals": journals,
               "journals_cross": journals_cross, "table": table})
