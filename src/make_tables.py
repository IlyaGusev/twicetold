"""LaTeX tables for the paper from the reports. Usage: make_tables.py <model>  -> paper/tables/*.tex"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS, TABLES = ROOT / "reports", ROOT / "paper" / "tables"
TABLES.mkdir(exist_ok=True)


def fmt(c, s, d=3, star=True):
    txt = f"{c:+.{d}f} ({s:.{d}f})"
    if star and s > 0 and abs(c / s) > 2.58:
        txt += "$^{*}$"
    return txt


def year_effects(model):
    e = pd.read_csv(REPORTS / f"year_effects_{model}.csv")
    rows = []
    specs = [("raw", "No controls (primary)"), ("full", "+ length, ratings, confidence, area"),
             ("unedited", "No edited review"), ("area-reweighted", "2017--22 area weights")]
    outs = [("S3", "$S_3$"), ("share_shared_claims", "Share shared"), ("hit_rate_strict", "Hit rate"), ("D3", "$D_3$")]
    lines = []
    for o, olab in outs:
        for sp, slab in specs:
            t = e[(e.conf == "ICLR") & (e.spec == sp) & (e.outcome == o)].set_index("year")
            if t.empty:
                continue
            cells = [fmt(t.loc[y, "coef"], t.loc[y, "se"]) if y in t.index else "" for y in [2024, 2025, 2026]]
            lines.append((olab if sp == "raw" else "", slab, *cells))
        for v, vlab in [("spec", "Specific claims only"), ("m8", "$\\le$8 claims per review")]:
            t = e[(e.conf == "ICLR") & (e.spec == "raw") & (e.outcome == f"{o}_{v}")].set_index("year")
            if t.empty:
                continue
            cells = [fmt(t.loc[y, "coef"], t.loc[y, "se"]) for y in [2024, 2025, 2026]]
            lines.append(("", vlab, *cells))
    out = ["\\begin{tabular}{llrrr}", "\\toprule", "Outcome & Specification & 2024 & 2025 & 2026 \\\\", "\\midrule"]
    prev = None
    for l in lines:
        if l[0] and prev is not None:
            out.append("\\addlinespace")
        out.append(" & ".join(l) + " \\\\")
        prev = l
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "year_effects.tex").write_text("\n".join(out) + "\n")


def within(model):
    w = pd.read_csv(REPORTS / f"within_paper_{model}.csv")
    p = pd.read_csv(REPORTS / f"pairs_within_{model}.csv")
    outs = [("share_shared", "Share of claims shared"), ("n_claims", "Claims"), ("n_unique", "Unique claims"), ("share_specific", "Specific share"),
            ("cat_clarity/presentation", "Clarity claims"), ("cat_experiments-scope", "Experiment-scope claims")]
    cols = [("ICLR", "2024-2026", "llm_flag"), ("ICLR", "2024-2026", "gz_isAI"), ("ICLR", "2024-2026", "llm_flag_ss"), ("NeurIPS", "2023-2025", "llm_flag"), ("ICLR", "2019-2022", "llm_flag")]
    out = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "& \\multicolumn{3}{c}{ICLR 2024--26} & NeurIPS 23--25 & ICLR 2019--22 \\\\",
           "Outcome of a review & Flag & GPTZero (2024) & Flag, S+S fields & Flag & Flag \\\\", "\\midrule"]
    for o, olab in outs:
        cells = []
        for conf, yrs, e in cols:
            t = w[(w.conf == conf) & (w.years == yrs) & (w.exposure == e) & (w.outcome == o)]
            cells.append(fmt(t.coef.iloc[0], t.se.iloc[0]) if len(t) else "--")
        out.append(olab + " & " + " & ".join(cells) + " \\\\")
    out += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Pairs of reviews of the same paper (hit rate; reference: neither flagged)}} \\\\",
            "Pair of reviews & All claims & Specific claims & ICLR 2026 & NeurIPS & \\\\"]
    for t, tlab in [("HL", "One review flagged"), ("LL", "Both reviews flagged")]:
        cells = []
        for conf, yrs, o in [("ICLR", "2024-2026", "hit"), ("ICLR", "2024-2026", "hit_spec"), ("ICLR", "2026-2026", "hit"), ("NeurIPS", "2023-2025", "hit"), (None, None, None)]:
            if conf is None:
                cells.append("--"); continue
            x = p[(p.conf == conf) & (p.years == yrs) & (p.outcome == o) & (p.pair_type == t)]
            cells.append(fmt(x.coef.iloc[0], x.se.iloc[0]) if len(x) else "--")
        out.append(tlab + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "within.tex").write_text("\n".join(out) + "\n")
    hh = p[(p.conf == "ICLR") & (p.years == "2024-2026") & (p.outcome == "hit") & (p.pair_type == "LL")].iloc[0]
    (TABLES / "pairs_note.tex").write_text(f"% HH mean hit {hh.mean_HH:.3f}; n pairs {hh.n}; n HH {hh.n_HH}; n LL {hh.n_t}\n")


def placebo(model):
    a = pd.read_csv(REPORTS / f"placebo_adjusted_{model}.csv")
    a = a[(a.placebo_kind == "same_area3") & (a.conf == "ICLR") & a.year.isin([2021, 2022, 2023, 2024, 2025, 2026])]
    out = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "ICLR & 2021 & 2022 & 2023 & 2024 & 2025 & 2026 \\\\", "\\midrule"]
    for o, olab in [("S3", "$S_3$ observed"), ("share_shared_claims", "Share shared observed")]:
        t = a[a.outcome == o].set_index("year")
        out.append(olab + " & " + " & ".join(f"{t.loc[y, 'observed']:.3f}" for y in t.index) + " \\\\")
        out.append("\\quad same-area placebo & " + " & ".join(f"{t.loc[y, 'placebo']:.3f}" for y in t.index) + " \\\\")
        out.append("\\quad excess & " + " & ".join(f"{t.loc[y, 'excess']:.3f} ({t.loc[y, 'se']:.3f})" for y in t.index) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "placebo.tex").write_text("\n".join(out) + "\n")


def e6(model):
    t = pd.read_csv(REPORTS / f"template_by_cell_{model}.csv")
    t = t[(t.conf == "ICLR") & t.year.isin([2019, 2021, 2023, 2024, 2025, 2026])].set_index("year")
    out = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "ICLR & 2019 & 2021 & 2023 & 2024 & 2025 & 2026 \\\\", "\\midrule"]
    for k, lab, d in [("hill_types_20k", "Effective number of types (of 300)", 0), ("tail_share", "Tail-type share", 3), ("template_share", "Template share", 3), ("specific", "Specific share", 3)]:
        out.append(lab + " & " + " & ".join(f"{t.loc[y, k]:.{d}f}" for y in t.index) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "types.tex").write_text("\n".join(out) + "\n")
    sh = pd.read_csv(REPORTS / f"type_shift_{model}.csv", index_col=0)
    out = ["\\begin{tabular}{lrr}", "\\toprule", "Critique type (label from pre-period examples) & 2023 & 2026 \\\\", "\\midrule", "\\multicolumn{3}{l}{\\emph{Largest falls}} \\\\"]
    for _, r in sh.head(8).iterrows():
        out.append(f"{r.short} & {100*r.f2023:.2f} & {100*r.f2026:.2f} \\\\")
    out.append("\\multicolumn{3}{l}{\\emph{Largest rises}} \\\\")
    for _, r in sh.tail(8).iloc[::-1].iterrows():
        out.append(f"{r.short} & {100*r.f2023:.2f} & {100*r.f2026:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "type_shift.tex").write_text("\n".join(out) + "\n")


def e5(model):
    L = pd.read_csv(REPORTS / f"llm_panels_pooled_{model}.csv", index_col=0).reindex(["hum4", "mix22", "llm4"])
    out = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Panel & Claims/review & $D_4$ & $S_4$ & Share shared & Hit rate & Specific share \\\\", "\\midrule"]
    for k, lab in [("hum4", "4 human reviews"), ("mix22", "2 human + 2 LLM"), ("llm4", "4 LLM reviews")]:
        r = L.loc[k]
        out.append(f"{lab} & {r.claims_per_review:.2f} & {r.D4:.1f} & {r.S4:.2f} & {r.share_shared_claims:.3f} & {r.hit_rate_strict:.3f} & {r.share_specific:.3f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "llm_panels.tex").write_text("\n".join(out) + "\n")


def variants(model):
    s = pd.read_csv(REPORTS / f"variant_sub3_{model}.csv", index_col=0)
    n = pd.read_csv(REPORTS / f"variant_noq_{model}.csv", index_col=0)
    out = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "ICLR & 2021 & 2022 & 2023 & 2024 & 2025 & 2026 \\\\", "\\midrule"]
    yrs = [2021, 2022, 2023, 2024, 2025, 2026]
    out.append("$S_3$, rarefied from the full panel & " + " & ".join(f"{s.loc[y, 'S3_rarefied']:.2f}" for y in yrs) + " \\\\")
    out.append("$S_3$, 3-review subset re-clustered & " + " & ".join(f"{s.loc[y, 'S3_recluster']:.2f} ({s.loc[y, 'se_S3']:.2f})" for y in yrs) + " \\\\")
    out.append("$S_3$, full text & " + " & ".join(["--"] * 3 + [f"{n.loc[y, 'S3_full']:.2f}" for y in yrs[3:]]) + " \\\\")
    out.append("$S_3$, questions field removed & " + " & ".join(["--"] * 3 + [f"{n.loc[y, 'S3_noq']:.2f} ({n.loc[y, 'se_S3']:.2f})" for y in yrs[3:]]) + " \\\\")
    out.append("Share shared, full text & " + " & ".join(["--"] * 3 + [f"{n.loc[y, 'share_shared_claims_full']:.3f}" for y in yrs[3:]]) + " \\\\")
    out.append("Share shared, questions removed & " + " & ".join(["--"] * 3 + [f"{n.loc[y, 'share_shared_claims_noq']:.3f}" for y in yrs[3:]]) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "variants.tex").write_text("\n".join(out) + "\n")


def categories(model):
    c = pd.read_csv(REPORTS / f"category_shift_{model}.csv", index_col=0)
    c = c.drop(index="other", errors="ignore")
    rows = list(c.head(5).index) + list(c.tail(6).index[::-1])
    short = {"clarity/presentation": "clarity, presentation", "typo/formatting": "typos, formatting", "theory/correctness": "theory, correctness",
             "related-work": "related work", "experiments-scope": "experiment scope", "compute-cost/scalability": "compute cost",
             "statistical-rigor": "statistical rigor", "reproducibility/details": "reproducibility"}
    out = ["\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lrrrr}", "\\toprule", "& & & \\multicolumn{2}{c}{2026 reviews} \\\\",
           "Category (\\% of claims) & 2021--23 & 2026 & unflagged & flagged \\\\", "\\midrule"]
    for k in rows:
        r = c.loc[k]
        out.append(f"{short.get(k, k)} & {r.pre21_23:.1f} & {r['2026']:.1f} & {r.unflagged_2026:.1f} & {r.flagged_2026:.1f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "categories.tex").write_text("\n".join(out) + "\n")


def validation(model):
    v = pd.read_csv(REPORTS / "pairs_validation.csv")
    out = ["\\begin{tabular}{lrrrr}", "\\toprule", "Pairs & $n$ & Precision & Recall & FPR \\\\", "\\midrule"]
    for sub, lab in [("all", "All"), ("pre", "ICLR 2017--22"), ("post", "ICLR 2026"), ("post, unflagged", "\\quad neither review flagged"), ("post, flagged", "\\quad a review flagged")]:
        r = v[(v.subset == sub) & (v.reference == "ref_problem")].iloc[0]
        out.append(f"{lab} & {int(r.n)} & {r.precision:.2f} & {r.recall:.2f} & {r.fpr:.3f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "pairs_validation.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    m = sys.argv[1]
    for f in [year_effects, within, placebo, e6, e5, variants, validation, categories]:
        f(m)
    print("tables written")
