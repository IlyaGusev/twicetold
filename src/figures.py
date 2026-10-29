"""Paper figures. Usage: figures.py <model>  -> paper/figs/*.pdf"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORTS, FIGS = ROOT / "data", ROOT / "reports", ROOT / "paper" / "figs"
FIGS.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.titlesize": 8, "legend.fontsize": 7, "legend.frameon": False})
C = {"ICLR": "#1f4e79", "NeurIPS": "#c0504d", "placebo": "#7f7f7f", "llm": "#e69f00", "pre": "#1f4e79", "mid": "#4a90c2", "post": "#c0504d"}


def ci(g):
    return g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(len(g))


def fig_trends(P, Q, model):
    outs = [("S3", "$S_3$: shared critiques per 3 reviews"), ("U3", "$U_3$: share raised by one reviewer"),
            ("D3", "$D_3$: distinct critiques per 3 reviews"), ("claims_per_review", "Claims per review")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85))
    for ax, (k, lab) in zip(axes, outs):
        for conf in ["ICLR", "NeurIPS"]:
            g = P[P.conf == conf].groupby("year")[k].apply(ci)
            years = g.index.to_numpy()
            m = np.array([v[0] for v in g]); e = np.array([v[1] for v in g])
            ax.errorbar(years, m, yerr=e, fmt="o-", ms=2.5, lw=1, color=C[conf], label=conf, capsize=1.5)
        if Q is not None and k in ("S3", "U3", "D3"):
            for kind, style, lab2 in [("same_area3", "s--", "same-area placebo"), ("shuffled4", "^:", "shuffled placebo")]:
                q = Q[(Q.conf == "ICLR") & (Q.placebo_kind == kind)]
                if len(q):
                    g = q.groupby("year")[k].mean()
                    ax.plot(g.index, g.values, style, ms=2, lw=0.8, color=C["placebo"], label=lab2)
        ax.axvspan(2022.9, 2026.3, color="#f0f0f0", zorder=0)
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([2017, 2020, 2023, 2026])
        ax.tick_params(labelsize=6.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(FIGS / f"trends_{model}.pdf", bbox_inches="tight")


def fig_rarefaction(P, model):
    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    groups = [("ICLR 2017-2022", P.year.between(2017, 2022), C["pre"]), ("ICLR 2024-2025", P.year.isin([2024, 2025]), C["mid"]),
              ("ICLR 2026", P.year == 2026, C["post"])]
    for name, mask, col in groups:
        sub = P[(P.conf == "ICLR") & mask & (P.n_reviews >= 4)]
        ks = [1, 2, 3, 4]
        m = [sub[f"D{k}"].mean() for k in ks]
        e = [1.96 * sub[f"D{k}"].std() / np.sqrt(len(sub)) for k in ks]
        ax.errorbar(ks, m, yerr=e, fmt="o-", ms=3, lw=1.2, color=col, label=f"{name} (n={len(sub)})", capsize=2)
    ax.plot([1, 4], [P[(P.conf == "ICLR") & P.year.between(2017, 2022) & (P.n_reviews >= 4)].D1.mean() * k for k in [1, 4]],
            ":", color="#999999", lw=0.8, label="no overlap (linear)")
    ax.set_xlabel("Number of reviews $k$")
    ax.set_ylabel("Expected distinct critiques $D_k$")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / f"rarefaction_{model}.pdf", bbox_inches="tight")


def fig_categories(model):
    T = pd.read_parquet(DATA / f"claims_table_{model}.parquet", columns=["conf", "year", "category"])
    T = T[T.conf == "ICLR"]
    f = pd.crosstab(T.category, T.year, normalize="columns") * 100
    pre = f[[2017, 2018, 2019, 2020, 2021, 2022, 2023]].mean(axis=1)
    d = pd.DataFrame(dict(pre=pre, y2025=f[2025], y2026=f[2026])).sort_values("pre")
    d = d.drop(index="other", errors="ignore")
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    y = np.arange(len(d))
    ax.barh(y + 0.25, d.pre, height=0.25, color=C["pre"], label="ICLR 2017-2023")
    ax.barh(y, d.y2025, height=0.25, color=C["mid"], label="ICLR 2025")
    ax.barh(y - 0.25, d.y2026, height=0.25, color=C["post"], label="ICLR 2026")
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=6.5)
    ax.set_xlabel("Share of all claims (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / f"categories_{model}.pdf", bbox_inches="tight")


def fig_within(model):
    path = REPORTS / f"within_paper_{model}.csv"
    if not path.exists():
        return
    W = pd.read_csv(path)
    W = W[(W.conf == "ICLR") & (W.years == "2024-2026")]
    exps = [e for e in ["llm_grade", "p_llm", "any_marker"] if e in set(W.exposure)]
    if not exps:
        return
    outs = ["share_shared", "n_unique", "share_specific", "cat_clarity/presentation", "cat_typo/formatting", "cat_ablation", "cat_generalization"]
    fig, axes = plt.subplots(1, len(exps), figsize=(2.3 * len(exps), 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, exps):
        g = W[W.exposure == e].set_index("outcome").reindex(outs)
        std = g.coef / g.sd_y
        err = 1.96 * g.se / g.sd_y
        ax.barh(np.arange(len(outs)), std, xerr=err, color=C["ICLR"], height=0.6, capsize=2)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(np.arange(len(outs)))
        ax.set_yticklabels([o.replace("cat_", "") for o in outs], fontsize=6.5)
        ax.set_title({"llm_grade": "Detector grade (0-3)", "p_llm": "Detector P(LLM)", "any_marker": "Marker words"}[e], fontsize=7)
        ax.set_xlabel("Within-paper effect (SD units)", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / f"within_{model}.pdf", bbox_inches="tight")


def fig_pairs(model):
    """Pair-level E3: hit rate of human-human, human-flagged, flagged-flagged review pairs of the same paper, by year,
    next to the all-LLM ceiling (E5)."""
    T = pd.read_parquet(DATA / f"pairs_outcomes_{model}.parquet")
    T = T[(T.conf == "ICLR") & T.year.between(2023, 2026)]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0), gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axes[0]
    for t, col, lab in [("HH", C["pre"], "neither flagged"), ("HL", C["mid"], "one flagged"), ("LL", C["post"], "both flagged")]:
        g = T[T.pair_type == t].groupby("year").hit.apply(ci)
        if g.empty:
            continue
        ax.errorbar(g.index, [v[0] for v in g], yerr=[v[1] for v in g], fmt="o-", ms=2.5, lw=1, color=col, label=lab, capsize=1.5)
    ax.set_xticks([2023, 2024, 2025, 2026])
    ax.set_ylabel("Pairwise hit rate")
    ax.set_title("Review pairs of the same paper, by detector flag", fontsize=7)
    ax.legend(loc="upper left")
    ax = axes[1]
    L = pd.read_csv(REPORTS / f"llm_panels_pooled_{model}.csv", index_col=0).reindex(["hum4", "mix22", "llm4"])
    x = np.arange(3)
    ax.bar(x - 0.18, L.hit_rate_strict, width=0.36, color=C["ICLR"], label="hit rate")
    ax.bar(x + 0.18, L.share_shared_claims, width=0.36, color=C["llm"], label="share of claims shared")
    ax.set_xticks(x)
    ax.set_xticklabels(["4 human", "2 human\n+2 LLM", "4 LLM"], fontsize=6.5)
    ax.set_title("Same papers, panels of LLM reviews (E5)", fontsize=7)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / f"pairs_{model}.pdf", bbox_inches="tight")


def fig_types(model):
    """E6: breadth of the critique repertoire by venue-year."""
    T = pd.read_csv(REPORTS / f"template_by_cell_{model}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7))
    for ax, (k, lab) in zip(axes, [("hill_types_20k", "Effective number of types"), ("tail_share", "Share in tail types")]):
        for conf in ["ICLR", "NeurIPS"]:
            g = T[T.conf == conf]
            ax.plot(g.year, g[k], "o-", ms=2.5, lw=1, color=C[conf], label=conf)
        ax.axvspan(2022.9, 2026.3, color="#f0f0f0", zorder=0)
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([2017, 2020, 2023, 2026])
        ax.tick_params(labelsize=6.5)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / f"types_{model}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    model = sys.argv[1]
    P = pd.read_parquet(DATA / f"panel_outcomes_{model}.parquet")
    P = P[P.n_claims > 0]
    pl = DATA / f"placebo_outcomes_{model}.parquet"
    Q = pd.read_parquet(pl) if pl.exists() else None
    fig_trends(P, Q, model)
    fig_rarefaction(P, model)
    if (DATA / f"claims_table_{model}.parquet").exists():
        fig_categories(model)
    fig_within(model)
    fig_pairs(model)
    fig_types(model)
    print("figures written to", FIGS)
