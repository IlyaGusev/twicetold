import json
from itertools import combinations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import assign, load_jsonl, panel_metrics
from settings import settings

BLIND_YEARS = [2018, 2022, 2024, 2026]
BLIND_PER_YEAR = 10
PAIRS_PER_STRATUM = 100
BLIND_OUTCOMES = ["n_claims", "n_clusters", "n_targets", "n_shared_clusters", "share_shared_claims",
                  "share_shared_claims_lenient", "D3", "S3", "U3", "T3", "pair_rate_strict", "hit_rate_strict",
                  "share_specific", "hill_shannon"]
VARIANT_COLS = ["S3", "D3", "U3", "share_shared_claims", "hit_rate_strict", "pair_rate_strict", "share_specific",
                "claims_per_review"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, items):
    with open(path, "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items)


def load_flags(data):
    return pd.read_parquet(Path(data) / "detector_scores.parquet", columns=["review_id", "llm_flag"]).set_index("review_id").llm_flag


def review_texts(data):
    return pd.read_parquet(Path(data) / "sample_reviews.parquet", columns=["review_id", "text"]).set_index("review_id").text


def model_claims(path):
    return {k: v["claims"] for k, v in load_jsonl(path, "review_id").items() if "claims" in v}


def blind_key(d):
    return json.loads((d / "blind_key.json").read_text())


def load_opus(d, ks):
    claims, panels = {}, {}
    for k in ks:
        claims.update({r["review_id"]: r["claims"] for r in read_jsonl(d / f"opus_claims_{k}.jsonl")})
        panels.update({p["panel_id"]: p for p in read_jsonl(d / f"opus_clusters_{k}.jsonl")})
    return claims, panels


def clustered(model, *groups):
    codes = [pd.factorize(g)[0] for g in groups]
    return model.fit(cov_type="cluster", cov_kwds={"groups": codes[0] if len(codes) == 1 else np.column_stack(codes)})


def write_split(items, path_fmt, k=4):
    for i in range(k):
        write_jsonl(str(path_fmt).format(i), items[i::k])


def blind_set(data=settings.DATA_DIR):
    data = Path(data)
    out = data / "blind"
    rng = np.random.default_rng(42)
    sel = pd.read_parquet(data / "sample_reviews.parquet", columns=["review_id", "paper_id", "conf", "year", "text"])
    sel = sel[(sel.conf == "ICLR") & sel.year.isin(BLIND_YEARS)]
    four = sel.groupby("paper_id").filter(lambda g: len(g) == 4)
    papers = [p for y in BLIND_YEARS for p in rng.choice(four[four.year == y].paper_id.unique(), BLIND_PER_YEAR, replace=False)]
    rng.shuffle(papers)

    out.mkdir(exist_ok=True)
    key, items = {}, []
    for i, p in enumerate(papers):
        g = four[four.paper_id == p]
        code = f"B{i:02d}"
        key[code] = {"paper_id": p, "year": int(g.year.iloc[0]), "review_ids": g.review_id.tolist()}
        items.append({"paper": code, "reviews": [{"review": f"R{j}", "text": t} for j, t in enumerate(g.text)]})
    (out / "blind_key.json").write_text(json.dumps(key, indent=1))
    write_split(items, out / "blind_input_{}.jsonl")
    write_jsonl(data / "panels_blind.jsonl", [{"panel_id": code, "review_ids": v["review_ids"]} for code, v in key.items()])
    print(len(items), "papers;", sum(len(r["text"]) for it in items for r in it["reviews"]), "chars")


def pairs_set(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    data = Path(data)
    out = data / "pairs"
    rng = np.random.default_rng(3)
    claims = model_claims(data / f"claims_{model}.jsonl")
    panels = load_jsonl(data / f"clusters_{model}.jsonl", "panel_id")
    meta = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    texts, flag = review_texts(data), load_flags(data)
    pools = {(per, s): [] for per in ("pre", "post_flag", "post_noflag") for s in ("same_cluster", "same_target", "different")}
    for pid, p in tqdm(panels.items(), desc="panels"):
        if pid not in meta.index or meta.loc[pid, "conf"] != "ICLR":
            continue
        year = int(meta.loc[pid, "year"])
        period = "pre" if year <= 2022 else "post" if year == 2026 else None
        if period is None:
            continue
        df = assign(p, claims)
        if df.empty:
            continue
        a = df.sample(frac=1, random_state=int(rng.integers(1e9)))
        seen = 0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                x, y = a.iloc[i], a.iloc[j]
                if x.review == y.review:
                    continue
                s = "same_cluster" if x.cluster == y.cluster else "same_target" if x.target == y.target else "different"
                if s == "different" and rng.random() > 0.02:
                    continue
                fa, fb = bool(flag.get(x.review_id, False)), bool(flag.get(y.review_id, False))
                key = period if period == "pre" else ("post_flag" if fa or fb else "post_noflag")
                pools[(key, s)].append({"paper_id": pid, "year": year, "stratum": s, "a_review": x.review_id, "b_review": y.review_id,
                                        "a_flag": fa, "b_flag": fb, "a_claim": x.claim, "b_claim": y.claim, "a_cat": x.category, "b_cat": y.category})
                seen += 1
                if seen > 30:
                    break
            if seen > 30:
                break
    rows = []
    for key, pool in pools.items():
        print(key, len(pool))
        n = PAIRS_PER_STRATUM if key[0] == "pre" else PAIRS_PER_STRATUM // 2
        idx = rng.choice(len(pool), min(n, len(pool)), replace=False)
        rows += [pool[i] for i in idx]
    rng.shuffle(rows)
    K = pd.DataFrame(rows)
    K["pair_id"] = [f"Q{i:03d}" for i in range(len(K))]
    out.mkdir(exist_ok=True)
    K.to_parquet(out / "pairs_key.parquet")
    items = [{"pair_id": r.pair_id, "claim_a": r.a_claim, "claim_b": r.b_claim,
              "review_a": texts.get(r.a_review, "")[:6000], "review_b": texts.get(r.b_review, "")[:6000]} for r in K.itertuples()]
    write_split(items, out / "pairs_input_{}.jsonl")
    print(len(K), "pairs;", K.groupby(["year", "stratum"]).size().to_dict())


def spec_set(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    data = Path(data)
    out = data / "spec"
    T = pd.read_parquet(data / f"claims_table_{model}.parquet", columns=["review_id", "k", "conf", "year", "claim", "specific"])
    T = T[T.conf == "ICLR"]
    pre = T[T.year.between(2017, 2022)].sample(150, random_state=11)
    post = T[T.year.between(2024, 2026)].sample(150, random_state=11)
    S = pd.concat([pre, post]).sample(frac=1, random_state=11).reset_index(drop=True)
    S["item_id"] = [f"C{i:03d}" for i in range(len(S))]
    out.mkdir(exist_ok=True)
    S.to_parquet(out / "spec_key.parquet")
    write_jsonl(out / "spec_input.jsonl", [{"item_id": r.item_id, "claim": r.claim} for r in S.itertuples()])
    print(len(S), "claims; pipeline specific share", round(S.specific.mean(), 3))


def opus_metrics(blind, ks=range(4)):
    claims, panels = load_opus(blind, [k for k in ks if (blind / f"opus_claims_{k}.jsonl").exists()])
    return pd.DataFrame([dict(panel_id=pid, **panel_metrics(assign(p, claims))) for pid, p in panels.items()]).set_index("panel_id")


def model_metrics(blind, data, model, tag):
    key = blind_key(blind)
    claims = model_claims(data / f"claims_{model}{tag}.jsonl")
    panels = load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id")
    return pd.DataFrame([dict(panel_id=code, year=v["year"], **panel_metrics(assign(panels[code], claims)))
                         for code, v in key.items() if code in panels]).set_index("panel_id")


def reference_agreement(blind):
    from scipy.stats import spearmanr

    both = opus_metrics(blind).join(opus_metrics(blind, ["rep0", "rep1"]), how="inner", suffixes=("_a", "_b"))
    if both.empty:
        return
    print(f"== Opus reference vs independent Opus replicate: {len(both)} papers")
    for k in BLIND_OUTCOMES:
        a, b = both[f"{k}_a"], both[f"{k}_b"]
        d = b - a
        print(f"{k:30s} {a.mean():7.3f} {b.mean():7.3f} rho {spearmanr(a, b).statistic:5.2f}  bias {d.mean():+.3f} (SE {d.std(ddof=1)/np.sqrt(len(d)):.3f})")


def blind(model=settings.DEFAULT_MODEL, tag="", data=settings.DATA_DIR):
    from scipy.stats import norm, spearmanr

    data = Path(data)
    bdir = data / "blind"
    reference_agreement(bdir)
    both = opus_metrics(bdir).join(model_metrics(bdir, data, model, tag), how="inner", suffixes=("_opus", "_model"))
    print(f"== {model}{tag} vs blind Opus reference: {len(both)} papers")
    print(f"{'outcome':30s} {'opus':>7s} {'model':>7s} {'rho':>6s}")
    for k in BLIND_OUTCOMES:
        a, b = both[f"{k}_opus"], both[f"{k}_model"]
        print(f"{k:30s} {a.mean():7.3f} {b.mean():7.3f} {spearmanr(a, b).statistic:6.2f}")
    print("\nBias (model - opus) by year, with SE:")
    for k in ["n_claims", "n_clusters", "n_shared_clusters", "share_shared_claims", "S3", "U3", "hit_rate_strict"]:
        d = both[f"{k}_model"] - both[f"{k}_opus"]
        print(f"  {k:22s}", {y: f"{v.mean():+.3f} ({v.std(ddof=1)/np.sqrt(len(v)):.3f})" for y, v in d.groupby(both.year)})
    print("\nEquivalence (TOST) of year bias: p that |bias(2026) - bias(2018-2022)| < bound")
    for k, bound in [("share_shared_claims", 0.03), ("S3", 0.3), ("hit_rate_strict", 0.03)]:
        d = both[f"{k}_model"] - both[f"{k}_opus"]
        late, early = d[both.year == 2026], d[both.year.isin([2018, 2022])]
        diff = late.mean() - early.mean()
        se = np.sqrt(late.var(ddof=1) / len(late) + early.var(ddof=1) / len(early))
        p = max(1 - norm.cdf((diff + bound) / se), 1 - norm.cdf((bound - diff) / se))
        print(f"  {k:22s} diff {diff:+.3f} (SE {se:.3f}), bound {bound}, TOST p={p:.3f}")
    print("\nMeans by year (opus | model) for S3 and share_shared_claims:")
    for k in ["S3", "share_shared_claims", "D3", "n_claims"]:
        print(f"  {k:22s}", both.groupby("year")[[f"{k}_opus", f"{k}_model"]].mean().round(3).to_dict("index"))


def load_judgments(pdir, run):
    rows = []
    for k in range(4):
        f = pdir / f"judgments_{run}{k}.jsonl"
        if not f.exists():
            continue
        ids = {r["pair_id"] for r in read_jsonl(pdir / f"pairs_input_{k}.jsonl")}
        js = read_jsonl(f)
        ok = [j for j in js if j["pair_id"] in ids]
        print(f"judgments_{run}{k}: {len(ok)} valid of {len(js)}")
        rows += ok
    return pd.DataFrame(rows).drop_duplicates("pair_id", keep="last").set_index("pair_id")


def pairs(data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    from sklearn.metrics import cohen_kappa_score

    pdir = Path(data) / "pairs"
    K = pd.read_parquet(pdir / "pairs_key.parquet").set_index("pair_id")
    K["period"] = np.where(K.year <= 2022, "pre", "post")
    K["flagged"] = K.a_flag | K.b_flag
    K["pipe_problem"] = K.stratum == "same_cluster"
    K["pipe_target"] = K.stratum != "different"
    both = K.join(load_judgments(pdir, "a").add_suffix("_a"), how="inner").join(load_judgments(pdir, "b").add_suffix("_b"), how="inner")
    print(f"{len(both)} pairs with both adjudications")
    for q in ["same_problem", "same_target"]:
        print(f"  run agreement on {q}: {(both[q+'_a'] == both[q+'_b']).mean():.3f}, kappa {cohen_kappa_score(both[q+'_a'], both[q+'_b']):.3f}")
    both["ref_problem"] = both.same_problem_a & both.same_problem_b
    both["ref_problem_any"] = both.same_problem_a | both.same_problem_b
    both["ref_target"] = both.same_target_a & both.same_target_b
    rows = []
    post = both.period == "post"
    for name, g in [("all", both), ("pre", both[~post]), ("post", both[post]),
                    ("post, unflagged", both[post & ~both.flagged]), ("post, flagged", both[post & both.flagged])]:
        for ref in ["ref_problem", "ref_problem_any"]:
            tp = (g.pipe_problem & g[ref]).sum()
            fp = (g.pipe_problem & ~g[ref]).sum()
            fn = (~g.pipe_problem & g[ref]).sum()
            tn = (~g.pipe_problem & ~g[ref]).sum()
            rows.append({"subset": name, "reference": ref, "n": len(g), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                         "fpr": fp / max(fp + tn, 1), "n_pipe_merge": int(g.pipe_problem.sum()), "n_ref_merge": int(g[ref].sum())})
    out = pd.DataFrame(rows)
    out.to_csv(Path(reports) / "pairs_validation.csv", index=False)
    print(out.round(3).to_string())
    print("\nPrecision of pipeline merges (both adjudicators agree) by stratum:")
    print(both[both.pipe_problem].groupby(["period", "flagged"]).ref_problem.agg(["mean", "size"]).round(3).to_string())
    print("\nRecall: share of adjudicated same-problem pairs that the pipeline merged, by stratum (sampling weights ignored):")
    print(both[both.ref_problem].groupby(["period", "flagged"]).pipe_problem.agg(["mean", "size"]).round(3).to_string())


def spec_block(d, name):
    from sklearn.metrics import cohen_kappa_score

    print(f"== {name} (n={len(d)})")
    print(f"  share specific: pipeline {d.pl.mean():.3f}, annotator A {d.ann_a.mean():.3f}, annotator B {d.ann_b.mean():.3f}")
    print(f"  kappa A vs B {cohen_kappa_score(d.ann_a, d.ann_b):.3f}; agreement {np.mean(d.ann_a == d.ann_b):.3f}")
    for a in "ab":
        print(f"  kappa pipeline vs {a.upper()} {cohen_kappa_score(d.pl, d[f'ann_{a}']):.3f}; agreement {np.mean(d.pl == d[f'ann_{a}']):.3f}")
    both = d[d.ann_a == d.ann_b]
    print(f"  on the {len(both)} items where A and B agree: pipeline accuracy {np.mean(both.pl == both.ann_a):.3f}, "
          f"precision {np.mean(both[both.pl].ann_a):.3f}, recall {np.mean(both[both.ann_a].pl):.3f}")


def spec(data=settings.DATA_DIR):
    sdir = Path(data) / "spec"
    K = pd.read_parquet(sdir / "spec_key.parquet").set_index("item_id")
    for a in "ab":
        J = {r["item_id"]: bool(r["specific"]) for r in read_jsonl(sdir / f"judgments_{a}.jsonl")}
        K[f"ann_{a}"] = K.index.map(J)
    K["pl"] = K.specific.astype(bool)
    K["period"] = np.where(K.year <= 2022, "2017-2022", "2024-2026")
    spec_block(K, "all")
    for p, d in K.groupby("period"):
        spec_block(d, p)


def ampere(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR):
    from scipy.stats import spearmanr

    from between import openai_embed

    data = Path(data)
    A = pd.read_parquet(data / "ampere_full.parquet").set_index("review_id")
    claims = {r["review_id"]: [c["claim"] for c in r["claims"]]
              for r in read_jsonl(data / f"claims_{model}_ampere.jsonl") if "claims" in r}
    A = A[A.index.isin(claims)]
    A["n_claims"] = A.index.map(lambda i: len(claims[i]))
    print(len(A), "reviews")
    for k in ["n_prop", "n_eval", "n_req"]:
        print(f"spearman(n_claims, {k}) = {spearmanr(A.n_claims, A[k]).statistic:.3f}")
    A["n_crit"] = A.n_eval + A.n_req
    print(f"spearman(n_claims, n_eval+n_req) = {spearmanr(A.n_claims, A.n_crit).statistic:.3f}; means claims {A.n_claims.mean():.1f}, eval+req {A.n_crit.mean():.1f}, req {A.n_req.mean():.1f}")

    cov, cov_eval, prec = [], [], []
    for rid, row in tqdm(A.iterrows(), total=len(A), desc="ampere reviews"):
        cl = claims[rid]
        if not cl:
            cov += [0] * row.n_req
            continue
        sents = list(row.sents)
        E = openai_embed(sents + cl, "text-embedding-3-small", batch=512, threads=1)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        sim = E[:len(sents)] @ E[len(sents):].T
        for s, lab in zip(sim.max(1), row.labels):
            if lab == "request":
                cov.append(s >= 0.5)
            if lab == "evaluation":
                cov_eval.append(s >= 0.5)
        arg = [lab in ("evaluation", "request") for lab in row.labels]
        prec += [arg[b] for b in sim.argmax(0)]
    print(f"request propositions covered by a claim: {np.mean(cov):.3f} (n={len(cov)})")
    print(f"evaluation propositions covered by a claim: {np.mean(cov_eval):.3f} (n={len(cov_eval)})")
    print(f"claims whose nearest proposition is evaluation/request: {np.mean(prec):.3f} (n={len(prec)})")


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def variants(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    data = Path(data)
    P = pd.read_parquet(data / f"panel_outcomes_{model}.parquet")
    P = P[P.n_claims > 0].set_index("panel_id")
    out = {}
    S = pd.read_parquet(data / f"sub3_outcomes_{model}.parquet")
    S = S[S.n_claims > 0].set_index("paper_id")
    j = S.join(P[["S3", "D3", "share_shared_claims", "hit_rate_strict"]], suffixes=("", "_full"))
    out["sub3"] = pd.DataFrame([{"year": y, "n": len(g), "S3_recluster": g.S3.mean(), "S3_rarefied": g.S3_full.mean(),
                                 "D3_recluster": g.D3.mean(), "D3_rarefied": g.D3_full.mean(),
                                 "share_recluster": g.share_shared_claims.mean(), "share_rarefied": g.share_shared_claims_full.mean(),
                                 "se_S3": se(g.S3)} for y, g in j.groupby("year")]).set_index("year")
    N = pd.read_parquet(data / f"panel_outcomes_{model}_noq.parquet")
    N = N[N.n_claims > 0].set_index("paper_id")
    j = N.join(P[VARIANT_COLS], suffixes=("", "_full"))
    cols = ["S3", "share_shared_claims", "claims_per_review", "D3"]
    out["noq"] = pd.DataFrame([dict(year=y, n=len(g), **{f"{c}_noq": g[c].mean() for c in cols},
                                    **{f"{c}_full": g[f"{c}_full"].mean() for c in cols}, se_S3=se(g.S3))
                               for y, g in j.groupby("year")]).set_index("year")
    W = pd.read_parquet(data / f"panel_outcomes_{model}_wd.parquet")
    W = W[W.n_claims > 0]
    nw = P[(P.conf == "ICLR") & (P.year == 2026)]
    out["wd"] = pd.DataFrame({"withdrawn": W[VARIANT_COLS].mean(), "withdrawn_se": W[VARIANT_COLS].agg(se),
                              "kept": nw[VARIANT_COLS].mean(), "kept_se": nw[VARIANT_COLS].agg(se)})
    out["wd"].loc["n"] = [len(W), np.nan, len(nw), np.nan]
    for k, v in out.items():
        v.round(4).to_csv(Path(reports) / f"variant_{k}_{model}.csv")
        print(f"== {k}\n{v.round(3).to_string()}\n")


def drift_set(data=settings.DATA_DIR, years=(2023, 2026), per_year=25, seed=7, k=5):
    data = Path(data)
    out = data / "blind2"
    rng = np.random.default_rng(seed)
    sol = pd.read_parquet(data / "panel_outcomes_gpt-6.1-sol.parquet", columns=["paper_id", "year", "n_reviews"])
    pool = sol[sol.year.isin(years) & (sol.n_reviews == 4)]
    papers = [p for y in years for p in rng.choice(pool[pool.year == y].paper_id, per_year, replace=False)]
    rng.shuffle(papers)
    panels = {p["panel_id"]: p for p in read_jsonl(data / "panels.jsonl")}
    text = review_texts(data)
    out.mkdir(exist_ok=True)
    key, items = {}, []
    for i, p in enumerate(papers):
        code, rids = f"C{i:02d}", panels[p]["review_ids"]
        key[code] = {"paper_id": p, "year": int(pool.set_index("paper_id").year[p]), "review_ids": rids}
        items.append({"paper": code, "reviews": [{"review": f"R{j}", "text": text[r]} for j, r in enumerate(rids)]})
    (out / "blind_key.json").write_text(json.dumps(key, indent=1))
    write_split(items, out / "blind_input_{}.jsonl", k)
    print(len(items), "papers")


def pair_audit_set(data=settings.DATA_DIR, year=2026, n=60, seed=9, k=5):
    data = Path(data)
    out = data / "blind5"
    rng = np.random.default_rng(seed)
    flag = load_flags(data)
    meta = pd.read_parquet(data / "panel_outcomes_gpt-6-luna.parquet", columns=["panel_id", "conf", "year", "n_reviews"])
    ids = set(meta[(meta.conf == "ICLR") & (meta.year == year) & (meta.n_reviews == 4)].panel_id)
    used = {v["paper_id"] for v in blind_key(data / "blind3").values()}
    panels = [p for p in read_jsonl(data / "panels.jsonl") if p["panel_id"] in ids and p["panel_id"] not in used
              and all(r in flag.index for r in p["review_ids"]) and flag[p["review_ids"]].sum() == 2]
    chosen = [panels[i] for i in rng.choice(len(panels), n, replace=False)]
    text = review_texts(data)
    pairs = []
    for p in chosen:
        for kind in (True, False):
            two = [r for r in p["review_ids"] if bool(flag[r]) == kind]
            pairs.append({"paper_id": p["panel_id"], "flagged": kind, "ids": list(rng.permutation(two))})
    rng.shuffle(pairs)
    out.mkdir(exist_ok=True)
    key, items = {}, []
    for i, q in enumerate(pairs):
        code = f"P{i:03d}"
        key[code] = q
        items.append({"pair": code, "A": text[q["ids"][0]], "B": text[q["ids"][1]]})
    (out / "pair_key.json").write_text(json.dumps(key, indent=1))
    write_split(items, out / "pair_input_{}.jsonl", k)
    print(f"{len(panels)} eligible papers, {n} chosen, {len(items)} pairs")


def pair_audit(data=settings.DATA_DIR, annotators=("x", "y"), reports=settings.REPORTS_DIR):
    from scipy.stats import pearsonr

    data = Path(data)
    annotators = annotators.split(",") if isinstance(annotators, str) else list(annotators)
    blind5 = data / "blind5"
    key = json.loads((blind5 / "pair_key.json").read_text())
    rates = {}
    for a in annotators:
        for path in sorted(blind5.glob(f"pair_out_{a}_*.jsonl")):
            for r in read_jsonl(path):
                na, nb = len(r["a_claims"]), len(r["b_claims"])
                ma, mb = len({m[0] for m in r["matches"]}), len({m[1] for m in r["matches"]})
                if na and nb:
                    rates.setdefault(r["pair"], {})[a] = (ma / na + mb / nb) / 2 * 100
    T = pd.DataFrame([{"pair": c, "paper": key[c]["paper_id"], "flagged": key[c]["flagged"], **v} for c, v in rates.items()])
    T = T.dropna(subset=annotators)
    T["rate"] = T[annotators].mean(axis=1)
    print(f"pairs {len(T)}; annotator agreement on pair match rate: r = {pearsonr(T[annotators[0]], T[annotators[1]])[0]:.2f}, "
          f"mean abs diff {np.abs(T[annotators[0]] - T[annotators[1]]).mean():.2f} points")
    W = T.pivot_table(index="paper", columns="flagged", values="rate").dropna()
    d = W[True] - W[False]
    print(f"papers {len(W)}: flagged pair {W[True].mean():.2f}, unflagged pair {W[False].mean():.2f}, "
          f"difference {d.mean():+.2f} (SE {d.sem():.2f}), relative {d.mean() / W[False].mean():+.0%}")
    for a in annotators:
        Wa = T.pivot_table(index="paper", columns="flagged", values=a).dropna()
        da = Wa[True] - Wa[False]
        print(f"  annotator {a}: difference {da.mean():+.2f} (SE {da.sem():.2f})")
    T.to_csv(Path(reports) / "pair_audit_blind5.csv", index=False)


def flag_set(data=settings.DATA_DIR, year=2026, n=50, seed=8, k=5):
    data = Path(data)
    out = data / "blind3"
    rng = np.random.default_rng(seed)
    flag = load_flags(data)
    meta = pd.read_parquet(data / "panel_outcomes_gpt-6-luna.parquet", columns=["panel_id", "conf", "year", "n_reviews"])
    ids = set(meta[(meta.conf == "ICLR") & (meta.year == year) & (meta.n_reviews == 4)].panel_id)
    panels = [p for p in read_jsonl(data / "panels.jsonl") if p["panel_id"] in ids
              and all(r in flag.index for r in p["review_ids"]) and flag[p["review_ids"]].sum() == 2]
    chosen = [panels[i] for i in rng.choice(len(panels), n, replace=False)]
    text = review_texts(data)
    out.mkdir(exist_ok=True)
    key, items = {}, []
    for i, p in enumerate(chosen):
        code, rids = f"F{i:02d}", p["review_ids"]
        key[code] = {"paper_id": p["panel_id"], "year": year, "review_ids": rids}
        items.append({"paper": code, "reviews": [{"review": f"R{j}", "text": text[r]} for j, r in enumerate(rids)]})
    (out / "blind_key.json").write_text(json.dumps(key, indent=1))
    write_split(items, out / "blind_input_{}.jsonl", k)
    print(f"{len(panels)} eligible papers, {len(items)} chosen")


def pair_hits(p, claims, real_ids):
    df = assign(p, claims)
    out = {}
    for a, b in combinations(range(len(p["review_ids"])), 2):
        ga, gb = df[df.review == a].cluster.to_numpy(), df[df.review == b].cluster.to_numpy()
        if len(ga) and len(gb):
            same = ga[:, None] == gb[None, :]
            out[(real_ids[a], real_ids[b])] = (same.any(1).mean() + same.any(0).mean()) / 2 * 100
    return out


def flags(data=settings.DATA_DIR, k=5, reports=settings.REPORTS_DIR):
    data = Path(data)
    blind3 = data / "blind3"
    key, flag = blind_key(blind3), load_flags(data)
    claims, panels = load_opus(blind3, range(k))
    sources = {"opus": {code: pair_hits(panels[code], claims, v["review_ids"]) for code, v in key.items() if code in panels}}
    for model in ["gpt-6-luna", "gpt-6.1-sol"]:
        mc = model_claims(data / f"claims_{model}.jsonl")
        mp = load_jsonl(data / f"clusters_{model}.jsonl", "panel_id")
        sources[model] = {code: pair_hits(mp[v["paper_id"]], mc, mp[v["paper_id"]]["review_ids"])
                          for code, v in key.items() if v["paper_id"] in mp}
    rows = []
    for src, by_paper in sources.items():
        for code, hits in by_paper.items():
            k_of = {pair: int(flag[pair[0]]) + int(flag[pair[1]]) for pair in hits}
            ll = [h for pair, h in hits.items() if k_of[pair] == 2]
            hh = [h for pair, h in hits.items() if k_of[pair] == 0]
            if ll and hh:
                rows.append({"source": src, "paper": code, "ll": np.mean(ll), "hh": np.mean(hh)})
    T = pd.DataFrame(rows)
    T["diff"] = T.ll - T.hh
    T.to_csv(Path(reports) / "flags_blind3.csv", index=False)
    print(T.groupby("source").agg(papers=("paper", "size"), ll=("ll", "mean"), hh=("hh", "mean"), diff=("diff", "mean"),
                                  se=("diff", "sem")).round(2).to_string())


def swap_detector(data=settings.DATA_DIR, k=5, reports=settings.REPORTS_DIR):
    import ai
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    data = Path(data)
    key = blind_key(data / "blind3")
    plan = json.loads((data / "blind4" / "swap_plan.json").read_text())
    stored = pd.read_parquet(data / "detector_scores.parquet", columns=["review_id", "p_llm", "llm_flag"]).set_index("review_id")
    q99 = stored[stored.llm_flag].p_llm.min()
    intel = pd.read_parquet(data / "intel_reviews.parquet", columns=["label", "year", "text"])
    train = intel[intel.year.isin([2021, 2022, 2023])]
    F = ai.style_features(train.text)
    tfidf = TfidfTransformer(sublinear_tf=True).fit(F)
    clf = LogisticRegression(C=4.0, max_iter=3000).fit(tfidf.transform(F), train.label)
    text = review_texts(data)
    rows = []
    for i in range(k):
        for p in read_jsonl(data / "blind4" / f"blind_input_{i}.jsonl"):
            code = "F" + p["paper"][1:]
            for j, r in enumerate(p["reviews"]):
                rid = key[code]["review_ids"][j]
                rows.append({"review_id": rid, "target": plan[code][f"R{j}"], "orig_text": text[rid], "swap_text": r["text"]})
    T = pd.DataFrame(rows)
    T["p_orig"] = clf.predict_proba(tfidf.transform(ai.style_features(T.orig_text)))[:, 1]
    T["p_swap"] = clf.predict_proba(tfidf.transform(ai.style_features(T.swap_text)))[:, 1]
    print("refit reproduces stored scores: max |diff|", float(np.abs(T.p_orig.values - stored.p_llm[T.review_id].values).max()))
    T["flag_orig"], T["flag_swap"] = T.p_orig >= q99, T.p_swap >= q99
    out = T.groupby("target").agg(n=("review_id", "size"), p_orig=("p_orig", "median"), p_swap=("p_swap", "median"),
                                  flag_orig=("flag_orig", "mean"), flag_swap=("flag_swap", "mean"))
    print(out.round(3).to_string())
    T.drop(columns=["orig_text", "swap_text"]).to_csv(Path(reports) / "swap_detector.csv", index=False)


def swap(data=settings.DATA_DIR, k=5, reports=settings.REPORTS_DIR):
    data = Path(data)
    key, flag = blind_key(data / "blind3"), load_flags(data)
    rows, sizes = [], []
    for name, d, prefix in [("original", data / "blind3", "F"), ("swapped", data / "blind4", "S")]:
        claims, panels = load_opus(d, range(k))
        for code in key:
            p = panels.get(prefix + code[1:])
            if p is None:
                continue
            real = key[code]["review_ids"]
            hits = pair_hits(p, claims, real)
            ll = [h for (a, b), h in hits.items() if flag[a] and flag[b]]
            hh = [h for (a, b), h in hits.items() if not flag[a] and not flag[b]]
            if ll and hh:
                rows.append({"version": name, "paper": code, "ll": np.mean(ll), "hh": np.mean(hh)})
            for j, r in enumerate(real):
                sizes.append({"version": name, "review": r, "flagged": bool(flag[r]), "claims": len(claims.get(p["review_ids"][j], []))})
    T = pd.DataFrame(rows)
    T["diff"] = T.ll - T.hh
    T.to_csv(Path(reports) / "swap_blind4.csv", index=False)
    print(T.groupby("version").agg(papers=("paper", "size"), ll=("ll", "mean"), hh=("hh", "mean"), diff=("diff", "mean"),
                                   se=("diff", "sem")).round(2).to_string())
    w = T.pivot(index="paper", columns="version", values="diff").dropna()
    d = w.swapped - w.original
    print(f"swapped - original gap, paired: {d.mean():+.2f} (SE {d.sem():.2f}), papers {len(w)}")
    S = pd.DataFrame(sizes).pivot_table(index=["review", "flagged"], columns="version", values="claims").dropna().reset_index()
    print("claims per review (original -> swapped):", S.groupby("flagged")[["original", "swapped"]].mean().round(2).to_dict("index"),
          f"| spearman {S.original.corr(S.swapped, method='spearman'):.2f}")


def drift(data=settings.DATA_DIR, k=5, reports=settings.REPORTS_DIR):
    data = Path(data)
    blind2 = data / "blind2"
    key = blind_key(blind2)
    ref = opus_metrics(blind2, range(k))[["hit_rate_strict", "share_shared_claims"]].add_suffix("_opus")
    rows = []
    for model in ["gpt-6-luna", "gpt-6.1-sol"]:
        P = pd.read_parquet(data / f"panel_outcomes_{model}.parquet", columns=["paper_id", "conf", "hit_rate_strict", "share_shared_claims"])
        P = P[P.conf == "ICLR"].set_index("paper_id")
        rows.append(P.loc[[v["paper_id"] for v in key.values()], ["hit_rate_strict", "share_shared_claims"]]
                    .set_axis(list(key), axis=0).add_suffix(f"_{model.split('-')[-1]}"))
    T = pd.concat([ref, *rows], axis=1).dropna()
    T["year"] = [key[c]["year"] for c in T.index]
    T.to_csv(Path(reports) / "drift_blind2.csv")
    for col in ["hit_rate_strict", "share_shared_claims"]:
        print(f"== {col} (x100), by year: mean (SE)")
        for src in ["opus", "luna", "sol"]:
            g = T.groupby("year")[f"{col}_{src}"]
            m, s = g.mean() * 100, g.sem() * 100
            print(f"  {src:5s}", {y: f"{m[y]:.2f} ({s[y]:.2f})" for y in m.index}, f"change {m.iloc[-1] - m.iloc[0]:+.2f} (SE {np.hypot(*s):.2f})")
        for src in ["luna", "sol"]:
            d = (T[f"{col}_{src}"] - T[f"{col}_opus"]) * 100
            g = d.groupby(T.year)
            print(f"  {src} - opus by year", g.mean().round(2).to_dict(),
                  f"drift {g.mean().iloc[-1] - g.mean().iloc[0]:+.2f} (SE {np.hypot(*g.sem()):.2f})")


def robust(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.REPORTS_DIR, tex=settings.TABLES_DIR / "robust.tex"):
    data = Path(data)
    P = pd.read_parquet(data / f"panel_outcomes_{model}.parquet").query("conf == 'ICLR' and n_claims > 0")
    pairs = pd.read_parquet(data / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "hit_spec"])
    spec = pairs[pairs.conf == "ICLR"].groupby(["panel_id", "year"]).hit_spec.mean().reset_index()
    sub3 = pd.read_parquet(data / f"sub3_outcomes_{model}.parquet").query("conf == 'ICLR' and n_claims > 0")
    noq = pd.read_parquet(data / f"panel_outcomes_{model}_noq.parquet").query("n_claims > 0")
    sol = pd.read_parquet(data / "panel_outcomes_gpt-6.1-sol.parquet").query("conf == 'ICLR' and n_claims > 0")
    wd = pd.read_parquet(data / f"panel_outcomes_{model}_wd.parquet").query("n_claims > 0")
    with_wd = pd.concat([P, wd.assign(year=2026)])
    same = P[P.paper_id.isin(sol.paper_id)]
    checks = [("Raw match rate", P, "hit_rate_strict", 2023), ("Specific claims only", spec, "hit_spec", 2023),
              ("Same part of paper (lenient)", P, "hit_rate_lenient", 2023),
              ("3-review subsets, grouped again", sub3, "hit_rate_strict", 2023),
              ("Questions field removed", noq, "hit_rate_strict", 2024),
              ("Withdrawn 2026 papers added", with_wd, "hit_rate_strict", 2023),
              ("Main model, second model's papers", same, "hit_rate_strict", 2023),
              ("Second model (gpt-6.1-sol)", sol, "hit_rate_strict", 2023),
              ("Main model, claims in shared groups", same, "share_shared_claims", 2023),
              ("Second model, claims in shared groups", sol, "share_shared_claims", 2023)]
    rows = []
    for name, df, col, base in checks:
        a, b = df[df.year == base][col].dropna() * 100, df[df.year == 2026][col].dropna() * 100
        rows.append({"check": name, "base_year": base, "base": a.mean(), "y2026": b.mean(), "change": b.mean() - a.mean(),
                     "se": np.hypot(se(a), se(b))})
    S = pd.read_csv(Path(reports) / f"match_std_{model}.csv").query("conf == 'ICLR'").set_index("year")
    rows.insert(1, {"check": "Standardized", "base_year": 2023, "base": S.loc[2023, "std"], "y2026": S.loc[2026, "std"],
                    "change": S.loc[2026, "std"] - S.loc[2023, "std"], "se": np.hypot(S.loc[2023, "std_se"], S.loc[2026, "std_se"])})
    out = pd.DataFrame(rows)
    out.round(3).to_csv(Path(reports) / f"robust_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Check & Base year & Base & 2026 & Change (SE) \\", r"\midrule"]
    lines += [f"{r.check} & {r.base_year} & {r.base:.1f} & {r.y2026:.1f} & {r.change:+.1f} ({r.se:.1f}) \\\\" for r in out.itertuples()]
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def fe_gap(P, flag, controls=()):
    import statsmodels.api as sm

    k = flag.reindex(P.a).to_numpy() + flag.reindex(P.b).to_numpy()
    X = pd.DataFrame({"one": (k == 1) * 1.0, "two": (k == 2) * 1.0, "hit": P.hit.to_numpy()}, index=P.index)
    X = X.join(P[list(controls)])
    X = X - X.groupby(P.panel_id.to_numpy()).transform("mean")
    fit = clustered(sm.OLS(X.hit, X[["one", "two", *controls]]), P.panel_id)
    return {f"{t}{s}": fit.params[t] * 100 if s == "" else fit.bse[t] * 100 for t in ["one", "two"] for s in ["", "_se"]}


def detector_placebo(model=settings.DEFAULT_MODEL, share=0.45, data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    from outcomes import ICLR_FROM

    data, reports = Path(data), Path(reports)
    P = pd.read_parquet(data / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    P = P[P.conf == "ICLR"]
    D = pd.read_parquet(data / "detector_scores.parquet", columns=["review_id", "p_llm", "llm_flag"]).set_index("review_id")
    trained = set(pd.read_parquet(data / "intel_reviews.parquet", columns=["paper_id", "year"]).query("year <= 2023").paper_id)
    periods = {f"{ICLR_FROM}": (ICLR_FROM, ICLR_FROM), "2021-2022": (2021, 2022), "2023": (2023, 2023),
               "2024-2026 (real flags)": (2024, 2026)}
    rows = []
    for name, (lo, hi) in tqdm(periods.items(), desc="periods"):
        S = P[P.year.between(lo, hi) & P.a.isin(D.index) & P.b.isin(D.index)]
        if lo >= 2024:
            flag = D.llm_flag.astype(int)
        else:
            ids = pd.Index(np.union1d(S.a, S.b))
            score = D.p_llm.reindex(ids)
            flag = (score >= score.quantile(1 - share)).astype(int)
        for subset, T in [("all", S), ("without detector-training papers", S[~S.panel_id.isin(trained)])]:
            rows.append(dict(period=name, subset=subset, papers=T.panel_id.nunique(), **fe_gap(T, flag)))
    out = pd.DataFrame(rows)
    out.to_csv(reports / f"detector_placebo_{model}.csv", index=False)
    print(out.round(2).to_string(index=False))


def rq1_evidence(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    from outcomes import pair_design

    data, reports = Path(data), Path(reports)
    flag = load_flags(data).astype(int)
    P = pair_design(model, data)
    n = pd.read_parquet(data / f"claims_table_{model}.parquet", columns=["review_id"]).groupby("review_id").size()
    na, nb = np.log(n.reindex(P.a).to_numpy()), np.log(n.reindex(P.b).to_numpy())
    P["claims_short"], P["claims_long"] = np.minimum(na, nb), np.maximum(na, nb)
    iclr = P[(P.conf == "ICLR") & (P.year >= 2024)].dropna()
    coco = pd.concat([pd.read_json(f, lines=True) for f in [data / "cocodet_main.jsonl", data / "cocodet_icml.jsonl"]])
    coco = (coco.set_index("review_id")[["human", "mix", "ai"]].idxmax(axis=1) != "human").astype(int)
    coco = coco[~coco.index.duplicated()]
    both_coco = iclr[iclr.a.isin(coco.index) & iclr.b.isin(coco.index)]
    sol = pd.read_parquet(data / "pairs_gpt-6.1-sol.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    sol = sol[(sol.conf == "ICLR") & (sol.year == 2026)]
    main_same = P[P.panel_id.isin(set(sol.panel_id)) & (P.year == 2026)]
    rows = []

    def add(group, label, est, se, papers):
        rows.append(dict(group=group, label=label, est=est, se=se, papers=papers))

    pipe = "Pipeline"
    for label, T, f, c in [("ICLR 2024–26", iclr, flag, ()),
                           ("+ length and claim count", iclr, flag, ("log_short", "log_long", "claims_short", "claims_long")),
                           ("NeurIPS 2024–25", P[(P.conf == "NeurIPS") & (P.year >= 2024)], flag, ()),
                           ("CoCoDet labels, ICLR", both_coco, coco, ()),
                           ("Main model, 500 ICLR 2026 papers", main_same, flag, ()),
                           ("Second model, same papers", sol, flag, ())]:
        g = fe_gap(T, f, c)
        add(pipe, label, g["two"], g["two_se"], T.panel_id.nunique())
    ref = "Blind Claude Opus"
    F = pd.read_csv(reports / "flags_blind3.csv").query("source == 'opus'")
    add(ref, "End to end, 50 papers", F["diff"].mean(), F["diff"].sem(), len(F))
    W = pd.read_csv(reports / "swap_blind4.csv").query("version == 'swapped'")
    add(ref, "After style swap, 48 papers", W["diff"].mean(), W["diff"].sem(), len(W))
    A = pd.read_csv(reports / "pair_audit_blind5.csv").pivot_table(index="paper", columns="flagged", values="rate")
    d = A[True] - A[False]
    add(ref, "Pair audit, 60 papers", d.mean(), d.sem(), len(d))
    Q = pd.read_csv(reports / f"detector_placebo_{model}.csv").query("subset == 'all' and not period.str.contains('real')")
    for r in Q.itertuples():
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
        for sa in root.iter("sub-article"):
            meta = sa.find("front-stub")
            meta = meta if meta is not None else sa.find("front/article-meta")
            if meta is None:
                continue
            n = meta.find(".//contrib/name")
            if n is not None:
                names[meta.findtext("article-id[@pub-id-type='doi']")] = f"{n.findtext('given-names')} {n.findtext('surname')}".lower()
    return pd.Series(names)


def reviewer_fe(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    import statsmodels.formula.api as smf
    from fetch import RAW

    data, reports = Path(data), Path(reports)
    name = f1000_reviewer_names(RAW / "f1000")
    P = pd.read_parquet(data / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    P = P[(P.conf == "F1000") & (P.year >= 2019) & (P.year != 2024)]
    L = pd.read_parquet(data / "other_reviews.parquet", columns=["review_id", "n_chars"]).set_index("review_id").n_chars
    R = pd.concat([P.rename(columns={"a": "rid", "b": "other"}), P.rename(columns={"b": "rid", "a": "other"})])
    R["log_len"], R["log_len_other"] = np.log(L.reindex(R.rid).to_numpy()), np.log(L.reindex(R.other).to_numpy())
    rv = R.groupby(["rid", "panel_id", "year"]).agg(hit=("hit", "mean"), log_len=("log_len", "first"),
                                                     log_len_other=("log_len_other", "mean"), size=("other", "size")).reset_index()
    rv["name"], rv["post"] = rv.rid.map(name), (rv.year >= 2025).astype(int)
    rv["hit"] *= 100
    span = rv.groupby("name").post.agg(["min", "max"])
    both = rv[rv.name.isin(span.index[(span["min"] == 0) & (span["max"] == 1)])]
    ctrl = " + log_len + log_len_other + C(size)"

    def fit(label, f, d):
        m = clustered(smf.ols(f, data=d), d.panel_id, d.name.fillna(d.rid))
        return dict(label=label, est=m.params["post"], se=m.bse["post"], reports=int(m.nobs), reviewers=d.name.nunique(),
                    pre=d[d.post == 0].hit.mean(), post=d[d.post == 1].hit.mean())

    out = pd.DataFrame([fit("All reports, pooled", "hit ~ post" + ctrl, rv),
                        fit("Reviewers seen in both periods, pooled", "hit ~ post" + ctrl, both),
                        fit("Reviewers seen in both periods, reviewer fixed effects", "hit ~ post + C(name)" + ctrl, both)])
    out.to_csv(reports / "reviewer_fe_f1000.csv", index=False)
    print(out.round(2).to_string(index=False))


def paper_side(model=settings.DEFAULT_MODEL, data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    import statsmodels.formula.api as smf
    from outcomes import STD_FORMULA, pair_design

    data, reports = Path(data), Path(reports)
    det = pd.read_json(data / "cocodet_abstracts.jsonl", lines=True).drop_duplicates("review_id").set_index("review_id")
    score = det.ai + det["mix"]
    P = pair_design(model, data)
    P = P[P.conf == "ICLR"].copy()
    P["score"] = score.reindex(P.panel_id).to_numpy()
    P = P.dropna(subset=["score"])
    P["z"] = (P.score - P.score.mean()) / P.score.std()
    flag = load_flags(data)
    P["nflag"] = flag.reindex(P.a).to_numpy().astype(float) + flag.reindex(P.b).to_numpy().astype(float)
    papers = P.groupby("year").panel_id.nunique()
    print(f"{P.panel_id.nunique()} ICLR papers with an abstract score; by year {papers.to_dict()}")
    print("share of abstracts CoCoDet labels AI or mixed:", P.groupby("year").score.apply(lambda s: (s > 0.5).mean()).round(3).to_dict())
    rows = []
    for subset, S in [("all pairs", P), ("neither flagged", P[P.nflag == 0]), ("both flagged", P[P.nflag == 2])]:
        m = clustered(smf.ols(STD_FORMULA + " + z", data=S), S.panel_id)
        rows.append(dict(kind="slope per SD of paper score", subset=subset, year="", est=m.params["z"] * 100, se=m.bse["z"] * 100,
                         papers=S.panel_id.nunique()))
    for subset, S in [("all pairs", P), ("neither flagged", P[P.nflag == 0])]:
        for kind, extra in [("standardized rate", ""), ("standardized rate + paper score", " + z + I(z**2)")]:
            fit = clustered(smf.ols(STD_FORMULA + extra, data=S), S.panel_id)
            names, V = pd.Index(fit.model.exog_names), fit.cov_params().to_numpy()
            is_year, x_mean = names.str.startswith("C(year)"), fit.model.exog.mean(0)
            for y in sorted(S.year.unique()):
                x = np.where(is_year, names == f"C(year)[T.{y}]", x_mean)
                rows.append(dict(kind=kind, subset=subset, year=y, est=x @ fit.params.to_numpy() * 100, se=np.sqrt(x @ V @ x) * 100,
                                 papers=S[S.year == y].panel_id.nunique()))
    out = pd.DataFrame(rows)
    out.to_csv(reports / "paper_side.csv", index=False)
    print(out[out.kind.str.startswith("slope")].round(2).to_string(index=False))
    print(out[~out.kind.str.startswith("slope")].pivot_table(index=["subset", "kind"], columns="year", values="est").round(2).to_string())


def instruments(models="gpt-6-luna,gpt-6.1-sol,glm-5.3-flash,deepseek-v4.1-flash", data=settings.DATA_DIR, reports=settings.REPORTS_DIR):
    from scipy.stats import spearmanr

    from outcomes import ICLR_FROM, panel_pairs, review_chars, year_rates

    data, reports = Path(data), Path(reports)
    models = models.split(",") if isinstance(models, str) else list(models)
    years = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "year"]).set_index("paper_id").year
    sol = {p["panel_id"]: len(p["review_ids"]) for f in ["panels_sol.jsonl", "panels_sol2.jsonl"] for p in read_jsonl(data / f)}
    flag = load_flags(data)
    opus = opus_metrics(data / "blind")
    runs = {m: (model_claims(data / f"claims_{m}.jsonl"), load_jsonl(data / f"clusters_{m}.jsonl", "panel_id")) for m in models}
    grouped = set.intersection(*(set(p) for _, p in runs.values()))
    common = sorted(grouped & set(sol))
    print(f"{len(common)} common ICLR panels")
    chars = review_chars(data, {r for pid in common for r in runs[models[0]][1][pid]["review_ids"]})
    rows, panel_rate = [], {}
    for m, (claims, panels) in runs.items():
        pr = []
        for pid in common:
            df = assign(panels[pid], claims)
            if not df.empty:
                pr.extend(dict(panel_id=pid, year=int(years[pid]), **r) for r in panel_pairs(df, panels[pid]["review_ids"]))
        P = pd.DataFrame(pr)
        pm = P.groupby(["year", "panel_id"]).hit.mean().reset_index()
        panel_rate[m] = pm.set_index("panel_id").hit
        n_claims = [len(claims.get(r, [])) for pid in common for r in panels[pid]["review_ids"]]
        rows.append(dict(model=m, stat="claims per review", est=np.mean(n_claims), se=se(pd.Series(n_claims)), n=len(n_claims)))
        for y, g in pm.groupby("year"):
            rows.append(dict(model=m, stat=f"match rate {y}", est=g.hit.mean() * 100, se=se(g.hit) * 100, n=len(g)))
        for base in [2023, 2025]:
            a, b = pm[pm.year == base].hit, pm[pm.year == 2026].hit
            rows.append(dict(model=m, stat=f"change {base}-2026", est=(b.mean() - a.mean()) * 100, se=np.hypot(se(a), se(b)) * 100, n=len(a) + len(b)))
        S = P[P.year >= ICLR_FROM].copy()
        la, lb = np.log(chars.reindex(S.a).to_numpy()), np.log(chars.reindex(S.b).to_numpy())
        S["log_short"], S["log_long"] = np.minimum(la, lb), np.maximum(la, lb)
        S["size"] = S.panel_id.map(sol).clip(upper=min(sol.values()) + 2)
        std = year_rates(S)
        for y in [2023, 2025, 2026]:
            rows.append(dict(model=m, stat=f"standardized rate {y}", est=std[y][0], se=std[y][1], n=int((S.year == y).sum())))
        for base in [2023, 2025]:
            rows.append(dict(model=m, stat=f"standardized change {base}-2026", est=std[2026][0] - std[base][0],
                             se=np.hypot(std[base][1], std[2026][1]), n=int(S.year.isin([base, 2026]).sum())))
        late = P[P.year >= 2024]
        gap = fe_gap(late, flag.astype(float))
        rows.append(dict(model=m, stat="gap: two flagged vs none, 2024-26", est=gap["two"], se=gap["two_se"], n=late.panel_id.nunique()))
        blind = [pid for pid in opus.index if pid in panels]
        if not blind:
            continue
        bm = pd.DataFrame([dict(panel_id=pid, **panel_metrics(assign(panels[pid], claims))) for pid in blind]).set_index("panel_id")
        both = opus.join(bm, how="inner", suffixes=("_opus", "_model"))
        d = both.hit_rate_strict_model - both.hit_rate_strict_opus
        rows.append(dict(model=m, stat="blind: bias vs Opus", est=d.mean() * 100, se=se(d) * 100, n=len(d)))
        rows.append(dict(model=m, stat="blind: rank agreement with Opus",
                         est=spearmanr(both.hit_rate_strict_opus, both.hit_rate_strict_model).statistic, se=np.nan, n=len(d)))
    for m in models[1:]:
        a, b = panel_rate[models[0]].align(panel_rate[m], join="inner")
        rows.append(dict(model=m, stat=f"rank agreement with {models[0]}", est=spearmanr(a, b).statistic, se=np.nan, n=len(a)))
    out = pd.DataFrame(rows)
    out.round(3).to_csv(reports / "instruments.csv", index=False)
    out["cell"] = out.apply(lambda r: f"{r.est:.2f}" if np.isnan(r.se) else f"{r.est:.1f} ({r.se:.1f})", axis=1)
    print(out.pivot(index="stat", columns="model", values="cell").reindex(out.stat.unique()).reindex(columns=models).to_string())


if __name__ == "__main__":
    fire.Fire({"blind_set": blind_set, "pairs_set": pairs_set, "spec_set": spec_set, "blind": blind, "pairs": pairs,
               "spec": spec, "ampere": ampere, "variants": variants, "robust": robust,
               "drift_set": drift_set, "drift": drift, "flag_set": flag_set, "flags": flags, "swap": swap, "swap_detector": swap_detector,
               "pair_audit_set": pair_audit_set, "pair_audit": pair_audit, "detector_placebo": detector_placebo, "rq1_evidence": rq1_evidence,
               "reviewer_fe": reviewer_fe, "paper_side": paper_side, "instruments": instruments})
