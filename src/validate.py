import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import assign, load_jsonl, overlap, panel_metrics
from settings import settings

BLIND_YEARS = [2018, 2022, 2024, 2026]
BLIND_PER_YEAR = 10
SHARDS = 4
PRE_LAST_YEAR = 2022
PAIRS_PER_STRATUM = 100
MAX_PAIRS_PER_PANEL = 31
DIFFERENT_KEEP_RATE = 0.02
PAIR_REVIEW_MAX_CHARS = 6000
PAIR_PERIODS = ("pre", "post_flag", "post_noflag")
PAIR_STRATA = ("same_cluster", "same_target", "different")
SPEC_PER_PERIOD = 150
SPEC_SEED = 11
COVER_MIN_SIM = 0.5
BLIND_OUTCOMES = [
    "n_claims",
    "n_clusters",
    "n_targets",
    "n_shared_clusters",
    "share_shared_claims",
    "share_shared_claims_lenient",
    "D3",
    "S3",
    "U3",
    "T3",
    "hit_rate_strict",
    "share_specific",
    "hill_shannon",
]
VARIANT_COLS = [
    "S3",
    "D3",
    "U3",
    "share_shared_claims",
    "hit_rate_strict",
    "share_specific",
    "claims_per_review",
]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, items):
    with open(path, "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items)


def load_flags(data):
    return (
        pd.read_parquet(Path(data) / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag
    )


def review_texts(data):
    return pd.read_parquet(Path(data) / "sample_reviews.parquet", columns=["review_id", "text"]).set_index("review_id").text


def model_claims(path):
    return {k: v["claims"] for k, v in load_jsonl(path, "review_id").items() if "claims" in v}


def blind_key(blind_dir):
    return json.loads((blind_dir / "blind_key.json").read_text())


def load_opus(blind_dir, ks):
    claims, panels = {}, {}
    for k in ks:
        claims.update({r["review_id"]: r["claims"] for r in read_jsonl(blind_dir / f"opus_claims_{k}.jsonl")})
        panels.update({p["panel_id"]: p for p in read_jsonl(blind_dir / f"opus_clusters_{k}.jsonl")})
    return claims, panels


def clustered(model, *groups):
    codes = [pd.factorize(g)[0] for g in groups]
    return model.fit(cov_type="cluster", cov_kwds={"groups": codes[0] if len(codes) == 1 else np.column_stack(codes)})


def write_split(items, path_fmt, k=SHARDS):
    for i in range(k):
        write_jsonl(str(path_fmt).format(i), items[i::k])


def se(x):
    return x.std(ddof=1) / np.sqrt(x.count())


def refuse_overwrite(key_path):
    if key_path.exists():
        raise SystemExit(f"{key_path} exists and its annotations are keyed to it; delete it by hand to draw a new set")


def draw_blind_set(reviews, out):
    rng = np.random.default_rng(42)
    reviews = reviews[(reviews.conf == "ICLR") & reviews.year.isin(BLIND_YEARS)]
    four_review = reviews.groupby("paper_id").filter(lambda g: len(g) == 4)
    papers = []
    for year in BLIND_YEARS:
        candidates = four_review[four_review.year == year].paper_id.unique()
        papers.extend(rng.choice(candidates, BLIND_PER_YEAR, replace=False))
    rng.shuffle(papers)

    out.mkdir(exist_ok=True)
    key, items = {}, []
    for i, paper_id in enumerate(papers):
        paper_reviews = four_review[four_review.paper_id == paper_id]
        code = f"B{i:02d}"
        key[code] = {
            "paper_id": paper_id,
            "year": int(paper_reviews.year.iloc[0]),
            "review_ids": paper_reviews.review_id.tolist(),
        }
        items.append({"paper": code, "reviews": [{"review": f"R{j}", "text": t} for j, t in enumerate(paper_reviews.text)]})
    write_blind(out, key, items, SHARDS)
    print(len(items), "papers;", sum(len(r["text"]) for it in items for r in it["reviews"]), "chars")


def blind_set(data=settings.DATA_DIR):
    data = Path(data)
    out = data / "blind"
    reviews = pd.read_parquet(data / "sample_reviews.parquet", columns=["review_id", "paper_id", "conf", "year", "text"])
    if (out / "blind_key.json").exists():
        print(f"keeping {out / 'blind_key.json'}: the Opus annotations are keyed to it")
    else:
        draw_blind_set(reviews, out)
    key = blind_key(out)
    review_ids = {r for v in key.values() for r in v["review_ids"]}
    blind_reviews = reviews[reviews.review_id.isin(review_ids)][["review_id", "text"]]
    if len(blind_reviews) != len(review_ids):
        raise SystemExit(f"sample_reviews.parquet lacks {len(review_ids) - len(blind_reviews)} reviews of the blind key")
    write_jsonl(data / "panels_blind.jsonl", [{"panel_id": code, "review_ids": v["review_ids"]} for code, v in key.items()])
    blind_reviews.to_parquet(data / "blind_reviews.parquet")


def pairs_set(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    data = Path(data)
    out = data / "pairs"
    refuse_overwrite(out / "pairs_key.parquet")
    rng = np.random.default_rng(3)
    claims = model_claims(data / f"claims_{model}.jsonl")
    panels = load_jsonl(data / f"clusters_{model}.jsonl", "panel_id")
    meta = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    texts = review_texts(data)
    flag = load_flags(data)
    pools = {(period, stratum): [] for period in PAIR_PERIODS for stratum in PAIR_STRATA}
    for pid, panel in tqdm(panels.items(), desc="panels"):
        if pid not in meta.index or meta.loc[pid, "conf"] != "ICLR":
            continue
        year = int(meta.loc[pid, "year"])
        if not (year <= PRE_LAST_YEAR or year == 2026):
            continue
        panel_claims = assign(panel, claims)
        if panel_claims.empty:
            continue
        shuffled = panel_claims.sample(frac=1, random_state=int(rng.integers(1e9)))
        seen = 0
        for x, y in combinations(shuffled.itertuples(index=False), 2):
            if x.review == y.review:
                continue
            if x.cluster == y.cluster:
                stratum = "same_cluster"
            elif x.target == y.target:
                stratum = "same_target"
            else:
                stratum = "different"
            if stratum == "different" and rng.random() > DIFFERENT_KEEP_RATE:
                continue
            a_flag, b_flag = bool(flag.get(x.review_id, False)), bool(flag.get(y.review_id, False))
            if year <= PRE_LAST_YEAR:
                period = "pre"
            else:
                period = "post_flag" if a_flag or b_flag else "post_noflag"
            pools[(period, stratum)].append(
                {
                    "paper_id": pid,
                    "year": year,
                    "stratum": stratum,
                    "a_review": x.review_id,
                    "b_review": y.review_id,
                    "a_flag": a_flag,
                    "b_flag": b_flag,
                    "a_claim": x.claim,
                    "b_claim": y.claim,
                    "a_cat": x.category,
                    "b_cat": y.category,
                }
            )
            seen += 1
            if seen == MAX_PAIRS_PER_PANEL:
                break
    rows = []
    for (period, stratum), pool in pools.items():
        print((period, stratum), len(pool))
        n = PAIRS_PER_STRATUM if period == "pre" else PAIRS_PER_STRATUM // 2
        picked = rng.choice(len(pool), min(n, len(pool)), replace=False)
        rows += [pool[i] for i in picked]
    rng.shuffle(rows)
    pairs_key = pd.DataFrame(rows)
    pairs_key["pair_id"] = [f"Q{i:03d}" for i in range(len(pairs_key))]
    out.mkdir(exist_ok=True)
    pairs_key.to_parquet(out / "pairs_key.parquet")
    items = [
        {
            "pair_id": r.pair_id,
            "claim_a": r.a_claim,
            "claim_b": r.b_claim,
            "review_a": texts.get(r.a_review, "")[:PAIR_REVIEW_MAX_CHARS],
            "review_b": texts.get(r.b_review, "")[:PAIR_REVIEW_MAX_CHARS],
        }
        for r in pairs_key.itertuples()
    ]
    write_split(items, out / "pairs_input_{}.jsonl")
    print(len(pairs_key), "pairs;", pairs_key.groupby(["year", "stratum"]).size().to_dict())


def spec_set(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    data = Path(data)
    out = data / "spec"
    refuse_overwrite(out / "spec_key.parquet")
    claims = pd.read_parquet(
        data / f"claims_table_{model}.parquet", columns=["review_id", "k", "conf", "year", "claim", "specific"]
    )
    claims = claims[claims.conf == "ICLR"]
    pre = claims[claims.year.between(2017, PRE_LAST_YEAR)].sample(SPEC_PER_PERIOD, random_state=SPEC_SEED)
    post = claims[claims.year.between(2024, 2026)].sample(SPEC_PER_PERIOD, random_state=SPEC_SEED)
    sample = pd.concat([pre, post]).sample(frac=1, random_state=SPEC_SEED).reset_index(drop=True)
    sample["item_id"] = [f"C{i:03d}" for i in range(len(sample))]
    out.mkdir(exist_ok=True)
    sample.to_parquet(out / "spec_key.parquet")
    write_jsonl(out / "spec_input.jsonl", [{"item_id": r.item_id, "claim": r.claim} for r in sample.itertuples()])
    print(len(sample), "claims; pipeline specific share", round(sample.specific.mean(), 3))


def opus_metrics(blind_dir, ks=range(SHARDS)):
    claims, panels = load_opus(blind_dir, [k for k in ks if (blind_dir / f"opus_claims_{k}.jsonl").exists()])
    rows = [panel_metrics(assign(p, claims), len(p["review_ids"])) for p in panels.values()]
    return pd.DataFrame(rows, index=pd.Index(list(panels), name="panel_id"))


def model_metrics(blind_dir, data, model, tag):
    key = blind_key(blind_dir)
    claims = model_claims(data / f"claims_{model}{tag}.jsonl")
    panels = load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id")
    return pd.DataFrame(
        [
            dict(panel_id=code, year=v["year"], **panel_metrics(assign(panels[code], claims), len(panels[code]["review_ids"])))
            for code, v in key.items()
            if code in panels
        ]
    ).set_index("panel_id")


def reference_agreement(blind_dir, reports=settings.ANALYSIS_REPORTS_DIR):
    from scipy.stats import spearmanr

    both = opus_metrics(blind_dir).join(opus_metrics(blind_dir, ["rep0", "rep1"]), how="inner", suffixes=("_a", "_b"))
    if both.empty:
        return
    both.reset_index().to_csv(Path(reports) / "blind_replicate.csv", index=False)
    print(f"== Opus reference vs independent Opus replicate: {len(both)} papers")
    for k in BLIND_OUTCOMES:
        a, b = both[f"{k}_a"], both[f"{k}_b"]
        bias = b - a
        print(
            f"{k:30s} {a.mean():7.3f} {b.mean():7.3f} rho {spearmanr(a, b).statistic:5.2f}  "
            f"bias {bias.mean():+.3f} (SE {se(bias):.3f})"
        )


def blind(model=settings.DEFAULT_MODEL, tag="", data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    from scipy.stats import norm, spearmanr

    data = Path(data)
    blind_dir = data / "blind"
    reference_agreement(blind_dir, reports)
    both = opus_metrics(blind_dir).join(model_metrics(blind_dir, data, model, tag), how="inner", suffixes=("_opus", "_model"))
    both.reset_index().to_csv(Path(reports) / f"blind_{model}.csv", index=False)
    print(f"== {model}{tag} vs blind Opus reference: {len(both)} papers")
    print(f"{'outcome':30s} {'opus':>7s} {'model':>7s} {'rho':>6s}")
    for k in BLIND_OUTCOMES:
        a, b = both[f"{k}_opus"], both[f"{k}_model"]
        print(f"{k:30s} {a.mean():7.3f} {b.mean():7.3f} {spearmanr(a, b).statistic:6.2f}")
    print("\nBias (model - opus) by year, with SE:")
    for k in ["n_claims", "n_clusters", "n_shared_clusters", "share_shared_claims", "S3", "U3", "hit_rate_strict"]:
        bias = both[f"{k}_model"] - both[f"{k}_opus"]
        print(f"  {k:22s}", {y: f"{v.mean():+.3f} ({se(v):.3f})" for y, v in bias.groupby(both.year)})
    print("\nEquivalence (TOST) of year bias: p that |bias(2026) - bias(2018-2022)| < bound")
    for k, bound in [("share_shared_claims", 0.03), ("S3", 0.3), ("hit_rate_strict", 0.03)]:
        bias = both[f"{k}_model"] - both[f"{k}_opus"]
        late, early = bias[both.year == 2026], bias[both.year.isin([2018, 2022])]
        diff = late.mean() - early.mean()
        diff_se = np.hypot(se(late), se(early))
        p = max(1 - norm.cdf((diff + bound) / diff_se), 1 - norm.cdf((bound - diff) / diff_se))
        print(f"  {k:22s} diff {diff:+.3f} (SE {diff_se:.3f}), bound {bound}, TOST p={p:.3f}")
    print("\nMeans by year (opus | model) for S3 and share_shared_claims:")
    for k in ["S3", "share_shared_claims", "D3", "n_claims"]:
        print(f"  {k:22s}", both.groupby("year")[[f"{k}_opus", f"{k}_model"]].mean().round(3).to_dict("index"))


def load_judgments(pdir, run):
    rows = []
    for k in range(SHARDS):
        path = pdir / f"judgments_{run}{k}.jsonl"
        if not path.exists():
            continue
        ids = {r["pair_id"] for r in read_jsonl(pdir / f"pairs_input_{k}.jsonl")}
        judgments = read_jsonl(path)
        valid = [j for j in judgments if j["pair_id"] in ids]
        print(f"judgments_{run}{k}: {len(valid)} valid of {len(judgments)}")
        rows += valid
    return pd.DataFrame(rows).drop_duplicates("pair_id", keep="last").set_index("pair_id")


def pairs(data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    from sklearn.metrics import cohen_kappa_score

    pdir = Path(data) / "pairs"
    pairs_key = pd.read_parquet(pdir / "pairs_key.parquet").set_index("pair_id")
    pairs_key["period"] = np.where(pairs_key.year <= PRE_LAST_YEAR, "pre", "post")
    pairs_key["flagged"] = pairs_key.a_flag | pairs_key.b_flag
    pairs_key["pipe_problem"] = pairs_key.stratum == "same_cluster"
    both = pairs_key.join(load_judgments(pdir, "a").add_suffix("_a"), how="inner").join(
        load_judgments(pdir, "b").add_suffix("_b"), how="inner"
    )
    print(f"{len(both)} pairs with both adjudications")
    for q in ["same_problem", "same_target"]:
        run_a, run_b = both[f"{q}_a"], both[f"{q}_b"]
        print(f"  run agreement on {q}: {(run_a == run_b).mean():.3f}, kappa {cohen_kappa_score(run_a, run_b):.3f}")
    both["ref_problem"] = both.same_problem_a & both.same_problem_b
    both["ref_problem_any"] = both.same_problem_a | both.same_problem_b
    both["ref_target"] = both.same_target_a & both.same_target_b
    rows = []
    post = both.period == "post"
    for name, subset in [
        ("all", both),
        ("pre", both[~post]),
        ("post", both[post]),
        ("post, unflagged", both[post & ~both.flagged]),
        ("post, flagged", both[post & both.flagged]),
    ]:
        for ref in ["ref_problem", "ref_problem_any"]:
            tp = (subset.pipe_problem & subset[ref]).sum()
            fp = (subset.pipe_problem & ~subset[ref]).sum()
            fn = (~subset.pipe_problem & subset[ref]).sum()
            tn = (~subset.pipe_problem & ~subset[ref]).sum()
            rows.append(
                {
                    "subset": name,
                    "reference": ref,
                    "n": len(subset),
                    "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(tp + fn, 1),
                    "fpr": fp / max(fp + tn, 1),
                    "n_pipe_merge": int(subset.pipe_problem.sum()),
                    "n_ref_merge": int(subset[ref].sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / "pairs_validation.csv", index=False)
    print(out.round(3).to_string())
    print("\nPrecision of pipeline merges (both adjudicators agree) by stratum:")
    print(both[both.pipe_problem].groupby(["period", "flagged"]).ref_problem.agg(["mean", "size"]).round(3).to_string())
    print("\nRecall: share of adjudicated same-problem pairs that the pipeline merged, by stratum (sampling weights ignored):")
    print(both[both.ref_problem].groupby(["period", "flagged"]).pipe_problem.agg(["mean", "size"]).round(3).to_string())


def spec_block(items, name):
    from sklearn.metrics import cohen_kappa_score

    print(f"== {name} (n={len(items)})")
    print(
        f"  share specific: pipeline {items.pipeline.mean():.3f}, "
        f"annotator A {items.ann_a.mean():.3f}, annotator B {items.ann_b.mean():.3f}"
    )
    kappa = cohen_kappa_score(items.ann_a, items.ann_b)
    print(f"  kappa A vs B {kappa:.3f}; agreement {np.mean(items.ann_a == items.ann_b):.3f}")
    for a in "ab":
        ann = items[f"ann_{a}"]
        kappa = cohen_kappa_score(items.pipeline, ann)
        print(f"  kappa pipeline vs {a.upper()} {kappa:.3f}; agreement {np.mean(items.pipeline == ann):.3f}")
    agreed = items[items.ann_a == items.ann_b]
    print(
        f"  on the {len(agreed)} items where A and B agree: pipeline accuracy {np.mean(agreed.pipeline == agreed.ann_a):.3f}, "
        f"precision {np.mean(agreed[agreed.pipeline].ann_a):.3f}, recall {np.mean(agreed[agreed.ann_a].pipeline):.3f}"
    )


def spec(data=settings.DATA_DIR):
    sdir = Path(data) / "spec"
    spec_key = pd.read_parquet(sdir / "spec_key.parquet").set_index("item_id")
    for a in "ab":
        judged = {r["item_id"]: bool(r["specific"]) for r in read_jsonl(sdir / f"judgments_{a}.jsonl")}
        spec_key[f"ann_{a}"] = spec_key.index.map(judged)
    spec_key["pipeline"] = spec_key.specific.astype(bool)
    spec_key["period"] = np.where(spec_key.year <= PRE_LAST_YEAR, "2017-2022", "2024-2026")
    spec_block(spec_key, "all")
    for period, items in spec_key.groupby("period"):
        spec_block(items, period)


def ampere(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    from scipy.stats import spearmanr

    from between import openai_embed

    data = Path(data)
    reviews = pd.read_parquet(data / "ampere_full.parquet").set_index("review_id")
    claims = {
        r["review_id"]: [c["claim"] for c in r["claims"]]
        for r in read_jsonl(data / f"claims_{model}_ampere.jsonl")
        if "claims" in r
    }
    reviews = reviews[reviews.index.isin(claims)]
    reviews["n_claims"] = reviews.index.map(lambda i: len(claims[i]))
    print(len(reviews), "reviews")
    for k in ["n_prop", "n_eval", "n_req"]:
        print(f"spearman(n_claims, {k}) = {spearmanr(reviews.n_claims, reviews[k]).statistic:.3f}")
    reviews["n_crit"] = reviews.n_eval + reviews.n_req
    print(
        f"spearman(n_claims, n_eval+n_req) = {spearmanr(reviews.n_claims, reviews.n_crit).statistic:.3f}; "
        f"means claims {reviews.n_claims.mean():.1f}, eval+req {reviews.n_crit.mean():.1f}, req {reviews.n_req.mean():.1f}"
    )

    covered_req, covered_eval, claim_on_argument = [], [], []
    for rid, row in tqdm(reviews.iterrows(), total=len(reviews), desc="ampere reviews"):
        review_claims = claims[rid]
        if not review_claims:
            covered_req += [False] * row.n_req
            covered_eval += [False] * row.n_eval
            continue
        sents = list(row.sents)
        emb = openai_embed(sents + review_claims, settings.EMBED_MODEL, batch=512, threads=1)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        sim = emb[: len(sents)] @ emb[len(sents) :].T
        for best, label in zip(sim.max(1), row.labels):
            if label == "request":
                covered_req.append(best >= COVER_MIN_SIM)
            if label == "evaluation":
                covered_eval.append(best >= COVER_MIN_SIM)
        is_argument = [label in ("evaluation", "request") for label in row.labels]
        claim_on_argument += [is_argument[nearest] for nearest in sim.argmax(0)]
    print(f"request propositions covered by a claim: {np.mean(covered_req):.3f} (n={len(covered_req)})")
    print(f"evaluation propositions covered by a claim: {np.mean(covered_eval):.3f} (n={len(covered_eval)})")
    print(
        f"claims whose nearest proposition is evaluation/request: {np.mean(claim_on_argument):.3f} (n={len(claim_on_argument)})"
    )


def nonempty_outcomes(path):
    outcomes = pd.read_parquet(path)
    return outcomes[outcomes.n_claims > 0]


def variants(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    data = Path(data)
    full = nonempty_outcomes(settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet").set_index("panel_id")
    out = {}
    sub3 = nonempty_outcomes(settings.ANALYSIS_DIR / f"sub3_outcomes_{model}.parquet").set_index("paper_id")
    sub3_joined = sub3.join(full[["S3", "D3", "share_shared_claims", "hit_rate_strict"]], suffixes=("", "_full"))
    sub3_joined = sub3_joined.dropna(subset=["S3", "S3_full"])
    out["sub3"] = pd.DataFrame(
        [
            {
                "year": y,
                "n": len(g),
                "S3_recluster": g.S3.mean(),
                "S3_rarefied": g.S3_full.mean(),
                "D3_recluster": g.D3.mean(),
                "D3_rarefied": g.D3_full.mean(),
                "share_recluster": g.share_shared_claims.mean(),
                "share_rarefied": g.share_shared_claims_full.mean(),
                "se_S3": se(g.S3),
            }
            for y, g in sub3_joined.groupby("year")
        ]
    ).set_index("year")
    noq = nonempty_outcomes(settings.ANALYSIS_DIR / f"panel_outcomes_{model}_noq.parquet").set_index("paper_id")
    noq_joined = noq.join(full[VARIANT_COLS], suffixes=("", "_full")).dropna(subset=["S3", "S3_full"])
    noq_cols = ["S3", "share_shared_claims", "claims_per_review", "D3"]
    out["noq"] = pd.DataFrame(
        [
            dict(
                year=y,
                n=len(g),
                **{f"{c}_noq": g[c].mean() for c in noq_cols},
                **{f"{c}_full": g[f"{c}_full"].mean() for c in noq_cols},
                se_S3=se(g.S3),
            )
            for y, g in noq_joined.groupby("year")
        ]
    ).set_index("year")
    withdrawn = nonempty_outcomes(settings.ANALYSIS_DIR / f"panel_outcomes_{model}_wd.parquet")
    kept = full[(full.conf == "ICLR") & (full.year == 2026)]
    out["wd"] = pd.DataFrame(
        {
            "withdrawn": withdrawn[VARIANT_COLS].mean(),
            "withdrawn_se": withdrawn[VARIANT_COLS].agg(se),
            "kept": kept[VARIANT_COLS].mean(),
            "kept_se": kept[VARIANT_COLS].agg(se),
        }
    )
    out["wd"].loc["n"] = [len(withdrawn), np.nan, len(kept), np.nan]
    for name, table in out.items():
        table.round(4).to_csv(Path(reports) / f"variant_{name}_{model}.csv")
        print(f"== {name}\n{table.round(3).to_string()}\n")


def blind_item(code, review_ids, text):
    return {"paper": code, "reviews": [{"review": f"R{j}", "text": text[r]} for j, r in enumerate(review_ids)]}


def write_blind(out, key, items, k):
    (out / "blind_key.json").write_text(json.dumps(key, indent=1))
    write_split(items, out / "blind_input_{}.jsonl", k)


def drift_set(data=settings.DATA_DIR, years=(2023, 2026), per_year=25, seed=7, k=5):
    data = Path(data)
    out = data / "blind2"
    refuse_overwrite(out / "blind_key.json")
    rng = np.random.default_rng(seed)
    second = pd.read_parquet(
        data / f"panel_outcomes_{settings.SECOND_MODEL}.parquet", columns=["paper_id", "year", "paper_n_reviews"]
    )
    pool = second[second.year.isin(years) & (second.paper_n_reviews == 4)]
    papers = []
    for year in years:
        papers.extend(rng.choice(pool[pool.year == year].paper_id, per_year, replace=False))
    rng.shuffle(papers)
    panels = {p["panel_id"]: p for p in read_jsonl(data / "panels.jsonl")}
    text = review_texts(data)
    out.mkdir(exist_ok=True)
    year_of = pool.set_index("paper_id").year
    key, items = {}, []
    for i, paper_id in enumerate(papers):
        code = f"C{i:02d}"
        review_ids = panels[paper_id]["review_ids"]
        key[code] = {"paper_id": paper_id, "year": int(year_of[paper_id]), "review_ids": review_ids}
        items.append(blind_item(code, review_ids, text))
    write_blind(out, key, items, k)
    print(len(items), "papers")


def iclr_four_review_panels(data, year):
    meta = pd.read_parquet(
        data / f"panel_outcomes_{settings.BLIND_SET_MODEL}.parquet", columns=["panel_id", "conf", "year", "paper_n_reviews"]
    )
    return set(meta[(meta.conf == "ICLR") & (meta.year == year) & (meta.paper_n_reviews == 4)].panel_id)


def two_flagged_panels(data, flag, panel_ids):
    return [
        p
        for p in read_jsonl(data / "panels.jsonl")
        if p["panel_id"] in panel_ids and all(r in flag.index for r in p["review_ids"]) and flag[p["review_ids"]].sum() == 2
    ]


def pair_audit_set(data=settings.DATA_DIR, year=2026, n=60, seed=9, k=5):
    data = Path(data)
    out = data / "blind5"
    refuse_overwrite(out / "pair_key.json")
    rng = np.random.default_rng(seed)
    flag = load_flags(data)
    candidates = iclr_four_review_panels(data, year)
    used = {v["paper_id"] for v in blind_key(data / "blind3").values()}
    panels = two_flagged_panels(data, flag, candidates - used)
    chosen = [panels[i] for i in rng.choice(len(panels), n, replace=False)]
    text = review_texts(data)
    audit_pairs = []
    for panel in chosen:
        for flagged in (True, False):
            same_flag = [r for r in panel["review_ids"] if bool(flag[r]) == flagged]
            audit_pairs.append({"paper_id": panel["panel_id"], "flagged": flagged, "ids": list(rng.permutation(same_flag))})
    rng.shuffle(audit_pairs)
    out.mkdir(exist_ok=True)
    key, items = {}, []
    for i, pair in enumerate(audit_pairs):
        code = f"P{i:03d}"
        key[code] = pair
        items.append({"pair": code, "A": text[pair["ids"][0]], "B": text[pair["ids"][1]]})
    (out / "pair_key.json").write_text(json.dumps(key, indent=1))
    write_split(items, out / "pair_input_{}.jsonl", k)
    print(f"{len(panels)} eligible papers, {n} chosen, {len(items)} pairs")


def as_list(value):
    return value.split(",") if isinstance(value, str) else list(value)


def audit_overlap(row):
    a_ids = [f"A{i + 1}" for i in range(len(row["a_claims"]))]
    b_ids = [f"B{i + 1}" for i in range(len(row["b_claims"]))]
    parent = {node: node for node in [*a_ids, *b_ids]}

    def root(node):
        while parent[node] != node:
            node = parent[node]
        return node

    for a, b in row["matches"]:
        parent[root(a)] = root(b)
    return overlap([root(node) for node in a_ids], [root(node) for node in b_ids])


def pair_audit(data=settings.DATA_DIR, annotators=("x", "y"), reports=settings.ANALYSIS_REPORTS_DIR):
    from scipy.stats import pearsonr

    data = Path(data)
    annotators = as_list(annotators)
    blind5 = data / "blind5"
    key = json.loads((blind5 / "pair_key.json").read_text())
    rates = {}
    for annotator in annotators:
        for path in sorted(blind5.glob(f"pair_out_{annotator}_*.jsonl")):
            for r in read_jsonl(path):
                if r["a_claims"] and r["b_claims"]:
                    rates.setdefault(r["pair"], {})[annotator] = audit_overlap(r) * 100
    audit = pd.DataFrame([{"pair": c, "paper": key[c]["paper_id"], "flagged": key[c]["flagged"], **v} for c, v in rates.items()])
    audit = audit.dropna(subset=annotators)
    audit["rate"] = audit[annotators].mean(axis=1)
    print(
        f"pairs {len(audit)}; annotator agreement on pair match rate: "
        f"r = {pearsonr(audit[annotators[0]], audit[annotators[1]])[0]:.2f}, "
        f"mean abs diff {np.abs(audit[annotators[0]] - audit[annotators[1]]).mean():.2f} points"
    )
    by_paper = audit.pivot_table(index="paper", columns="flagged", values="rate").dropna()
    gap = by_paper[True] - by_paper[False]
    print(
        f"papers {len(by_paper)}: flagged pair {by_paper[True].mean():.2f}, unflagged pair {by_paper[False].mean():.2f}, "
        f"difference {gap.mean():+.2f} (SE {gap.sem():.2f}), relative {gap.mean() / by_paper[False].mean():+.0%}"
    )
    for annotator in annotators:
        annotator_by_paper = audit.pivot_table(index="paper", columns="flagged", values=annotator).dropna()
        annotator_gap = annotator_by_paper[True] - annotator_by_paper[False]
        print(f"  annotator {annotator}: difference {annotator_gap.mean():+.2f} (SE {annotator_gap.sem():.2f})")
    audit.to_csv(Path(reports) / "pair_audit_blind5.csv", index=False)


def flag_set(data=settings.DATA_DIR, year=2026, n=50, seed=8, k=5):
    data = Path(data)
    out = data / "blind3"
    refuse_overwrite(out / "blind_key.json")
    rng = np.random.default_rng(seed)
    flag = load_flags(data)
    panels = two_flagged_panels(data, flag, iclr_four_review_panels(data, year))
    chosen = [panels[i] for i in rng.choice(len(panels), n, replace=False)]
    text = review_texts(data)
    out.mkdir(exist_ok=True)
    key, items = {}, []
    for i, panel in enumerate(chosen):
        code = f"F{i:02d}"
        review_ids = panel["review_ids"]
        key[code] = {"paper_id": panel["panel_id"], "year": year, "review_ids": review_ids}
        items.append(blind_item(code, review_ids, text))
    write_blind(out, key, items, k)
    print(f"{len(panels)} eligible papers, {len(items)} chosen")


def pair_hits(panel, claims, real_ids):
    from outcomes import hit

    df = assign(panel, claims)
    hits = {}
    for a, b in combinations(range(len(panel["review_ids"])), 2):
        clusters_a = df[df.review == a].cluster.to_numpy()
        clusters_b = df[df.review == b].cluster.to_numpy()
        if len(clusters_a) and len(clusters_b):
            hits[(real_ids[a], real_ids[b])] = hit(clusters_a, clusters_b) * 100
    return hits


def gap_summary(rows, by, path):
    gaps = pd.DataFrame(rows)
    gaps["diff"] = gaps.ll - gaps.hh
    gaps.to_csv(path, index=False)
    print(
        gaps.groupby(by)
        .agg(papers=("paper", "size"), ll=("ll", "mean"), hh=("hh", "mean"), diff=("diff", "mean"), se=("diff", "sem"))
        .round(2)
        .to_string()
    )
    return gaps


def flags(data=settings.DATA_DIR, k=5, reports=settings.ANALYSIS_REPORTS_DIR, model=settings.DEFAULT_MODEL):
    data = Path(data)
    blind3 = data / "blind3"
    key = blind_key(blind3)
    flag = load_flags(data)
    claims, panels = load_opus(blind3, range(k))
    opus_hits = {code: pair_hits(panels[code], claims, v["review_ids"]) for code, v in key.items() if code in panels}
    sources = {"opus": opus_hits}
    claims_by_review = model_claims(data / f"claims_{model}.jsonl")
    model_panels = load_jsonl(data / f"clusters_{model}.jsonl", "panel_id")
    sources[model] = {}
    for code, v in key.items():
        if v["paper_id"] in model_panels:
            panel = model_panels[v["paper_id"]]
            sources[model][code] = pair_hits(panel, claims_by_review, panel["review_ids"])
    rows = []
    for source, by_paper in sources.items():
        for code, hits in by_paper.items():
            n_flagged = {pair: int(flag[pair[0]]) + int(flag[pair[1]]) for pair in hits}
            both_flagged = [h for pair, h in hits.items() if n_flagged[pair] == 2]
            neither_flagged = [h for pair, h in hits.items() if n_flagged[pair] == 0]
            if both_flagged and neither_flagged:
                rows.append({"source": source, "paper": code, "ll": np.mean(both_flagged), "hh": np.mean(neither_flagged)})
    gap_summary(rows, "source", Path(reports) / "flags_blind3.csv")


def swap_detector(data=settings.DATA_DIR, k=5, reports=settings.ANALYSIS_REPORTS_DIR):
    import ai

    data = Path(data)
    key = blind_key(data / "blind3")
    plan = json.loads((data / "blind4" / "swap_plan.json").read_text())
    stored = pd.read_parquet(data / "detector_scores.parquet", columns=["review_id", "p_llm", "llm_flag"]).set_index("review_id")
    flag_threshold = stored[stored.llm_flag].p_llm.min()
    intel = pd.read_parquet(data / "intel_reviews.parquet", columns=["label", "year", "text"])
    clf, tfidf = ai.fit_detector(intel[intel.year.isin(ai.DETECTOR_TRAIN_YEARS)])
    text = review_texts(data)
    rows = []
    for i in range(k):
        for item in read_jsonl(data / "blind4" / f"blind_input_{i}.jsonl"):
            code = "F" + item["paper"][1:]
            for j, review in enumerate(item["reviews"]):
                review_id = key[code]["review_ids"][j]
                rows.append(
                    {
                        "review_id": review_id,
                        "target": plan[code][f"R{j}"],
                        "orig_text": text[review_id],
                        "swap_text": review["text"],
                    }
                )
    scored = pd.DataFrame(rows)
    scored["p_orig"] = clf.predict_proba(tfidf.transform(ai.style_features(scored.orig_text)))[:, 1]
    scored["p_swap"] = clf.predict_proba(tfidf.transform(ai.style_features(scored.swap_text)))[:, 1]
    refit_diff = np.abs(scored.p_orig.values - stored.p_llm[scored.review_id].values).max()
    print("refit reproduces stored scores: max |diff|", float(refit_diff))
    scored["flag_orig"] = scored.p_orig >= flag_threshold
    scored["flag_swap"] = scored.p_swap >= flag_threshold
    by_target = scored.groupby("target").agg(
        n=("review_id", "size"),
        p_orig=("p_orig", "median"),
        p_swap=("p_swap", "median"),
        flag_orig=("flag_orig", "mean"),
        flag_swap=("flag_swap", "mean"),
    )
    print(by_target.round(3).to_string())
    scored.drop(columns=["orig_text", "swap_text"]).to_csv(Path(reports) / "swap_detector.csv", index=False)


def swap(data=settings.DATA_DIR, k=5, reports=settings.ANALYSIS_REPORTS_DIR):
    data = Path(data)
    key = blind_key(data / "blind3")
    flag = load_flags(data)
    rows, sizes = [], []
    for version, blind_dir, prefix in [("original", data / "blind3", "F"), ("swapped", data / "blind4", "S")]:
        claims, panels = load_opus(blind_dir, range(k))
        for code in key:
            panel = panels.get(prefix + code[1:])
            if panel is None:
                continue
            real_ids = key[code]["review_ids"]
            hits = pair_hits(panel, claims, real_ids)
            both_flagged = [h for (a, b), h in hits.items() if flag[a] and flag[b]]
            neither_flagged = [h for (a, b), h in hits.items() if not flag[a] and not flag[b]]
            if both_flagged and neither_flagged:
                rows.append({"version": version, "paper": code, "ll": np.mean(both_flagged), "hh": np.mean(neither_flagged)})
            for j, review_id in enumerate(real_ids):
                sizes.append(
                    {
                        "version": version,
                        "review": review_id,
                        "flagged": bool(flag[review_id]),
                        "claims": len(claims.get(panel["review_ids"][j], [])),
                    }
                )
    gaps = gap_summary(rows, "version", Path(reports) / "swap_blind4.csv")
    by_paper = gaps.pivot(index="paper", columns="version", values="diff").dropna()
    change = by_paper.swapped - by_paper.original
    print(f"swapped - original gap, paired: {change.mean():+.2f} (SE {change.sem():.2f}), papers {len(by_paper)}")
    claim_counts = pd.DataFrame(sizes).pivot_table(index=["review", "flagged"], columns="version", values="claims")
    claim_counts = claim_counts.dropna().reset_index()
    print(
        "claims per review (original -> swapped):",
        claim_counts.groupby("flagged")[["original", "swapped"]].mean().round(2).to_dict("index"),
        f"| spearman {claim_counts.original.corr(claim_counts.swapped, method='spearman'):.2f}",
    )


def drift(data=settings.DATA_DIR, k=5, reports=settings.ANALYSIS_REPORTS_DIR, model=settings.DEFAULT_MODEL):
    data = Path(data)
    blind2 = data / "blind2"
    key = blind_key(blind2)
    cols = ["hit_rate_strict", "share_shared_claims"]
    ref = opus_metrics(blind2, range(k))[cols].add_suffix("_opus")
    model_outcomes = pd.read_parquet(
        settings.ANALYSIS_DIR / f"panel_outcomes_{model}.parquet", columns=["paper_id", "conf", *cols]
    )
    model_outcomes = model_outcomes[model_outcomes.conf == "ICLR"].set_index("paper_id")
    blind_rows = model_outcomes.reindex([v["paper_id"] for v in key.values()])[cols].set_axis(list(key), axis=0)
    table = pd.concat([ref, blind_rows.add_suffix(f"_{model}")], axis=1).dropna()
    table["year"] = [key[c]["year"] for c in table.index]
    table.to_csv(Path(reports) / "drift_blind2.csv")
    print("papers by year", table.year.value_counts().sort_index().to_dict())
    for col in cols:
        print(f"== {col} (x100), by year: mean (SE)")
        for src in ["opus", model]:
            by_year = table.groupby("year")[f"{col}_{src}"]
            mean, sem = by_year.mean() * 100, by_year.sem() * 100
            print(
                f"  {src:5s}",
                {y: f"{mean[y]:.2f} ({sem[y]:.2f})" for y in mean.index},
                f"change {mean.iloc[-1] - mean.iloc[0]:+.2f} (SE {np.hypot(*sem):.2f})",
            )
        gap = (table[f"{col}_{model}"] - table[f"{col}_opus"]) * 100
        gap_by_year = gap.groupby(table.year)
        mean = gap_by_year.mean()
        print(
            f"  {model} - opus by year",
            mean.round(2).to_dict(),
            f"drift {mean.iloc[-1] - mean.iloc[0]:+.2f} (SE {np.hypot(*gap_by_year.sem()):.2f})",
        )


def robust(
    model=settings.DEFAULT_MODEL,
    data=settings.DATA_DIR,
    reports=settings.ANALYSIS_REPORTS_DIR,
    tex=settings.TABLES_DIR / "robust.tex",
):
    data = Path(data)
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
        ("Main overlap rate", main, "hit_rate_strict", 2023),
        ("Specific claims only", specific, "hit_spec", 2023),
        ("Same part of paper (lenient)", main, "hit_rate_lenient", 2023),
        ("3-review subsets, grouped again", sub3, "hit_rate_strict", 2023),
        ("Questions field removed", noq, "hit_rate_strict", 2024),
        ("Withdrawn 2026 papers added", with_wd, "hit_rate_strict", 2023),
    ]
    rows = []
    for name, df, col, base_year in checks:
        base = df[df.year == base_year][col].dropna() * 100
        last = df[df.year == 2026][col].dropna() * 100
        rows.append(
            {
                "check": name,
                "base_year": base_year,
                "base": base.mean(),
                "y2026": last.mean(),
                "change": last.mean() - base.mean(),
                "se": np.hypot(se(base), se(last)),
            }
        )
    out = pd.DataFrame(rows)
    out.round(3).to_csv(Path(reports) / f"robust_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Check & Base year & Base & 2026 & Change (SE) \\", r"\midrule"]
    lines += [
        f"{r.check} & {r.base_year} & {r.base:.1f} & {r.y2026:.1f} & {r.change:+.1f} ({r.se:.1f}) \\\\" for r in out.itertuples()
    ]
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def fe_gap(design, flag, controls=()):
    import statsmodels.api as sm

    n_flagged = flag.reindex(design.a).to_numpy() + flag.reindex(design.b).to_numpy()
    X = pd.DataFrame(
        {"one": (n_flagged == 1) * 1.0, "two": (n_flagged == 2) * 1.0, "hit": design.hit.to_numpy()}, index=design.index
    )
    X = X.join(design[list(controls)])
    X = X - X.groupby(design.panel_id.to_numpy()).transform("mean")
    fit = clustered(sm.OLS(X.hit, X[["one", "two", *controls]]), design.panel_id)
    return {
        "one": fit.params["one"] * 100,
        "one_se": fit.bse["one"] * 100,
        "two": fit.params["two"] * 100,
        "two_se": fit.bse["two"] * 100,
    }


def detector_placebo(model=settings.DEFAULT_MODEL, share=0.45, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    from outcomes import ICLR_FROM

    data, reports = Path(data), Path(reports)
    iclr_pairs = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    iclr_pairs = iclr_pairs[iclr_pairs.conf == "ICLR"]
    scores = pd.read_parquet(data / "detector_scores.parquet", columns=["review_id", "p_llm", "llm_flag"]).set_index("review_id")
    trained = set(pd.read_parquet(data / "intel_reviews.parquet", columns=["paper_id", "year"]).query("year <= 2023").paper_id)
    periods = {
        str(ICLR_FROM): (ICLR_FROM, ICLR_FROM),
        "2021-2022": (2021, 2022),
        "2023": (2023, 2023),
        "2024-2026 (real flags)": (2024, 2026),
    }
    both_scored = iclr_pairs.a.isin(scores.index) & iclr_pairs.b.isin(scores.index)
    rows = []
    for name, (lo, hi) in tqdm(periods.items(), desc="periods"):
        period_pairs = iclr_pairs[iclr_pairs.year.between(lo, hi) & both_scored]
        has_real_flags = lo >= 2024
        if has_real_flags:
            flag = scores.llm_flag.astype(int)
        else:
            reviews = pd.Index(np.union1d(period_pairs.a, period_pairs.b))
            score = scores.p_llm.reindex(reviews)
            flag = (score >= score.quantile(1 - share)).astype(int)
        untrained = period_pairs[~period_pairs.panel_id.isin(trained)]
        for subset, subset_pairs in [("all", period_pairs), ("without detector-training papers", untrained)]:
            rows.append(dict(period=name, subset=subset, papers=subset_pairs.panel_id.nunique(), **fe_gap(subset_pairs, flag)))
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"detector_placebo_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def short_long_logs(values, design):
    log_a = np.log(values.reindex(design.a).to_numpy())
    log_b = np.log(values.reindex(design.b).to_numpy())
    return np.minimum(log_a, log_b), np.maximum(log_a, log_b)


def rq1_evidence(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    from outcomes import pair_design

    data, reports = Path(data), Path(reports)
    flag = load_flags(data).astype(int)
    design = pair_design(model, data)
    claim_counts = pd.read_parquet(data / f"claims_table_{model}.parquet", columns=["review_id"]).groupby("review_id").size()
    design["claims_short"], design["claims_long"] = short_long_logs(claim_counts, design)
    iclr = design[(design.conf == "ICLR") & (design.year >= 2024)].dropna()
    coco_raw = pd.concat([pd.read_json(f, lines=True) for f in [data / "cocodet_main.jsonl", data / "cocodet_icml.jsonl"]])
    coco_flag = (coco_raw.set_index("review_id")[["human", "mix", "ai"]].idxmax(axis=1) != "human").astype(int)
    coco_flag = coco_flag[~coco_flag.index.duplicated()]
    both_coco = iclr[iclr.a.isin(coco_flag.index) & iclr.b.isin(coco_flag.index)]
    rows = []

    def add(group, label, est, err, papers):
        rows.append(dict(group=group, label=label, est=est, se=err, papers=papers))

    pipe = "Pipeline"
    for label, subset_pairs, pair_flags, controls in [
        ("ICLR 2024–26", iclr, flag, ()),
        ("+ length and claim count", iclr, flag, ("log_short", "log_long", "claims_short", "claims_long")),
        ("NeurIPS 2024–25", design[(design.conf == "NeurIPS") & (design.year >= 2024)], flag, ()),
        ("CoCoDet labels, ICLR", both_coco, coco_flag, ()),
    ]:
        gap = fe_gap(subset_pairs, pair_flags, controls)
        add(pipe, label, gap["two"], gap["two_se"], subset_pairs.panel_id.nunique())
    ref = "Blind Claude Opus"
    opus_flags = pd.read_csv(reports / "flags_blind3.csv").query("source == 'opus'")
    add(ref, "End to end, 50 papers", opus_flags["diff"].mean(), opus_flags["diff"].sem(), len(opus_flags))
    opus_swap = pd.read_csv(reports / "swap_blind4.csv").query("version == 'swapped'")
    add(ref, "After style swap, 48 papers", opus_swap["diff"].mean(), opus_swap["diff"].sem(), len(opus_swap))
    audit_by_paper = pd.read_csv(reports / "pair_audit_blind5.csv").pivot_table(index="paper", columns="flagged", values="rate")
    audit_gap = (audit_by_paper[True] - audit_by_paper[False]).dropna()
    add(ref, "Pair audit, 60 papers", audit_gap.mean(), audit_gap.sem(), len(audit_gap))
    placebo = pd.read_csv(reports / f"detector_placebo_{model}.csv").query("subset == 'all' and not period.str.contains('real')")
    for r in placebo.itertuples():
        add("Placebo before ChatGPT", f"Top-45% scores, ICLR {r.period}", r.two, r.two_se, r.papers)
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"rq1_evidence_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def f1000_reviewer_names(xml_dir):
    import xml.etree.ElementTree as ET

    names = {}
    for f in tqdm(sorted(Path(xml_dir).glob("*.xml")), desc="f1000 xml"):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        for sub in root.iter("sub-article"):
            meta = sub.find("front-stub")
            if meta is None:
                meta = sub.find("front/article-meta")
            if meta is None:
                continue
            person = meta.find(".//contrib/name")
            if person is not None:
                full_name = f"{person.findtext('given-names')} {person.findtext('surname')}"
                names[meta.findtext("article-id[@pub-id-type='doi']")] = full_name.lower()
    return pd.Series(names)


def reviewer_fe(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    import statsmodels.formula.api as smf

    data, reports = Path(data), Path(reports)
    reviewer_name = f1000_reviewer_names(settings.RAW_JOURNAL_REVIEWS_DIR / "f1000")
    f1000 = pd.read_parquet(
        settings.ANALYSIS_DIR / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"]
    )
    f1000 = f1000[(f1000.conf == "F1000") & (f1000.year >= 2019) & (f1000.year != 2024)]
    sides = pd.concat([f1000.rename(columns={"a": "rid", "b": "other"}), f1000.rename(columns={"b": "rid", "a": "other"})])
    per_review = sides.groupby(["rid", "panel_id", "year"], as_index=False).agg(hit=("hit", "mean"))
    per_review["name"] = per_review.rid.map(reviewer_name)
    per_review["post"] = (per_review.year >= 2025).astype(int)
    per_review["hit"] *= 100
    span = per_review.groupby("name").post.agg(["min", "max"])
    seen_both = per_review[per_review.name.isin(span.index[(span["min"] == 0) & (span["max"] == 1)])]

    def fit(label, formula, df):
        result = clustered(smf.ols(formula, data=df), df.panel_id, df.name.fillna(df.rid))
        return dict(
            label=label,
            est=result.params["post"],
            se=result.bse["post"],
            reports=int(result.nobs),
            reviewers=df.name.nunique(),
            pre=df[df.post == 0].hit.mean(),
            post=df[df.post == 1].hit.mean(),
        )

    out = pd.DataFrame(
        [
            fit("All reports, pooled", "hit ~ post", per_review),
            fit("Reviewers seen in both periods, pooled", "hit ~ post", seen_both),
            fit("Reviewers seen in both periods, reviewer fixed effects", "hit ~ post + C(name)", seen_both),
        ]
    )
    out.to_csv(reports / f"reviewer_fe_f1000_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def paper_side(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.ANALYSIS_REPORTS_DIR):
    import statsmodels.formula.api as smf

    from outcomes import pair_design

    data, reports = Path(data), Path(reports)
    abstract_labels = (
        pd.read_json(data / "cocodet_abstracts.jsonl", lines=True).drop_duplicates("review_id").set_index("review_id")
    )
    score = abstract_labels["ai"] + abstract_labels["mix"]
    design = pair_design(model, data)
    design = design[design.conf == "ICLR"]
    design["score"] = score.reindex(design.panel_id).to_numpy()
    design = design.dropna(subset=["score"])
    paper_scores = design.drop_duplicates("panel_id").score
    design["z"] = (design.score - paper_scores.mean()) / paper_scores.std()
    flag = load_flags(data)
    design["nflag"] = flag.reindex(design.a).to_numpy().astype(float) + flag.reindex(design.b).to_numpy().astype(float)
    papers = design.groupby("year").panel_id.nunique()
    print(f"{design.panel_id.nunique()} ICLR papers with an abstract score; by year {papers.to_dict()}")
    abstracts = design.drop_duplicates("panel_id")
    print(
        "share of abstracts CoCoDet labels AI or mixed:",
        abstracts.groupby("year").score.apply(lambda s: (s > 0.5).mean()).round(3).to_dict(),
    )
    neither_flagged = design[design.nflag == 0]
    both_flagged = design[design.nflag == 2]
    rows = []
    for subset, subset_pairs in [("all pairs", design), ("neither flagged", neither_flagged), ("both flagged", both_flagged)]:
        paper_rates = subset_pairs.groupby(["panel_id", "year"], as_index=False).agg(hit=("hit", "mean"), z=("z", "first"))
        fit = clustered(smf.ols("hit ~ C(year) + z", data=paper_rates), paper_rates.panel_id)
        rows.append(dict(subset=subset, est=fit.params["z"] * 100, se=fit.bse["z"] * 100, papers=len(paper_rates)))
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"paper_side_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def estimate_cell(row):
    if np.isnan(row.se):
        return f"{row.est:.2f}"
    return f"{row.est:.1f} ({row.se:.1f})"


def instruments(
    model=settings.DEFAULT_MODEL, models=settings.INSTRUMENT_MODELS, data=settings.DATA_DIR, reports=settings.REPORTS_DIR
):
    from scipy.stats import spearmanr

    from outcomes import ICLR_FROM, main_sample_review_ids, panel_pairs, review_chars, year_rates

    data, reports = Path(data), Path(reports)
    models = list(dict.fromkeys([model, *as_list(models)]))
    years = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "year"]).set_index("paper_id").year
    panel_size = {
        p["panel_id"]: len(p["review_ids"]) for f in ["panels_sol.jsonl", "panels_sol2.jsonl"] for p in read_jsonl(data / f)
    }
    flag = load_flags(data)
    opus = opus_metrics(data / "blind")
    runs = {m: (model_claims(data / f"claims_{m}.jsonl"), load_jsonl(data / f"clusters_{m}.jsonl", "panel_id")) for m in models}
    grouped = set.intersection(*(set(panels) for _, panels in runs.values()))
    main_reviews = main_sample_review_ids(data)
    common = sorted(
        pid
        for pid in grouped & set(panel_size)
        if all(main_reviews.issuperset(panels[pid]["review_ids"]) for _, panels in runs.values())
    )
    print(f"{len(common)} common ICLR panels")
    first_panels = runs[models[0]][1]
    chars = review_chars(data, {r for pid in common for r in first_panels[pid]["review_ids"]})
    rows, panel_rate = [], {}

    def add(model, stat, est, err, n):
        rows.append(dict(model=model, stat=stat, est=est, se=err, n=n))

    for model, (claims, panels) in runs.items():
        pair_rows = []
        for pid in common:
            df = assign(panels[pid], claims)
            if not df.empty:
                pair_rows.extend(
                    dict(panel_id=pid, year=int(years[pid]), **r) for r in panel_pairs(df, panels[pid]["review_ids"])
                )
        model_pairs = pd.DataFrame(pair_rows)
        panel_hits = model_pairs.groupby(["year", "panel_id"]).hit.mean().reset_index()
        panel_rate[model] = panel_hits.set_index("panel_id").hit
        n_claims = [len(claims.get(r, [])) for pid in common for r in panels[pid]["review_ids"]]
        add(model, "claims per review", np.mean(n_claims), se(pd.Series(n_claims)), len(n_claims))
        for y, g in panel_hits.groupby("year"):
            add(model, f"match rate {y}", g.hit.mean() * 100, se(g.hit) * 100, len(g))
        for base in [2023, 2025]:
            base_hits = panel_hits[panel_hits.year == base].hit
            last_hits = panel_hits[panel_hits.year == 2026].hit
            change = (last_hits.mean() - base_hits.mean()) * 100
            change_se = np.hypot(se(base_hits), se(last_hits)) * 100
            add(model, f"change {base}-2026", change, change_se, len(base_hits) + len(last_hits))
        recent = model_pairs[model_pairs.year >= ICLR_FROM]
        recent["log_short"], recent["log_long"] = short_long_logs(chars, recent)
        recent["size"] = recent.panel_id.map(panel_size).clip(upper=min(panel_size.values()) + 2)
        std_rates = year_rates(recent)
        for y in [2023, 2025, 2026]:
            est, err = std_rates[y]
            add(model, f"standardized rate {y}", est, err, int((recent.year == y).sum()))
        for base in [2023, 2025]:
            base_est, base_err = std_rates[base]
            last_est, last_err = std_rates[2026]
            n_pairs = int(recent.year.isin([base, 2026]).sum())
            add(model, f"standardized change {base}-2026", last_est - base_est, np.hypot(base_err, last_err), n_pairs)
        late = model_pairs[model_pairs.year >= 2024]
        gap = fe_gap(late, flag.astype(float))
        add(model, "gap: two flagged vs none, 2024-26", gap["two"], gap["two_se"], late.panel_id.nunique())
        blind_ids = [pid for pid in opus.index if pid in panels]
        if not blind_ids:
            continue
        blind_metrics = pd.DataFrame(
            [
                dict(panel_id=pid, **panel_metrics(assign(panels[pid], claims), len(panels[pid]["review_ids"])))
                for pid in blind_ids
            ]
        )
        both = opus.join(blind_metrics.set_index("panel_id"), how="inner", suffixes=("_opus", "_model"))
        bias = both.hit_rate_strict_model - both.hit_rate_strict_opus
        add(model, "blind: bias vs Opus", bias.mean() * 100, se(bias) * 100, len(bias))
        rank = spearmanr(both.hit_rate_strict_opus, both.hit_rate_strict_model).statistic
        add(model, "blind: rank agreement with Opus", rank, np.nan, len(bias))
    for model in models[1:]:
        a, b = panel_rate[models[0]].align(panel_rate[model], join="inner")
        add(model, f"rank agreement with {models[0]}", spearmanr(a, b).statistic, np.nan, len(a))
    out = pd.DataFrame(rows)
    out.round(3).to_csv(reports / f"instruments_{models[0]}.csv", index=False)
    out["cell"] = out.apply(estimate_cell, axis=1)
    print(out.pivot(index="stat", columns="model", values="cell").reindex(out.stat.unique()).reindex(columns=models).to_string())


if __name__ == "__main__":
    fire.Fire(
        {
            "blind_set": blind_set,
            "pairs_set": pairs_set,
            "spec_set": spec_set,
            "blind": blind,
            "pairs": pairs,
            "spec": spec,
            "ampere": ampere,
            "variants": variants,
            "robust": robust,
            "drift_set": drift_set,
            "drift": drift,
            "flag_set": flag_set,
            "flags": flags,
            "swap": swap,
            "swap_detector": swap_detector,
            "pair_audit_set": pair_audit_set,
            "pair_audit": pair_audit,
            "detector_placebo": detector_placebo,
            "rq1_evidence": rq1_evidence,
            "reviewer_fe": reviewer_fe,
            "paper_side": paper_side,
            "instruments": instruments,
        }
    )
