import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from extract import CATEGORIES
from metrics import assign, load_jsonl, panel_metrics
from settings import settings

PANEL_FILES = ["panels.jsonl", "panels_icml.jsonl", "panels_other.jsonl"]
WITHDRAWN_TAG = "_wd"
DROP_REASON = "a review of a withdrawn or desk-rejected paper, or a review not in reviews.parquet"
MIN_PLACEBO_REVIEWS = 2
PLACEBO_DROP_REASON = f"fewer than {MIN_PLACEBO_REVIEWS} reviews left after removing reviews of withdrawn or desk-rejected papers"
DROP_REASONS = {"panel": DROP_REASON, "sub3": DROP_REASON, "placebo": PLACEBO_DROP_REASON}
SOURCE_FILES = {"_icml": "icml_sample.parquet", "_other": "other_reviews.parquet"}
SOURCE_PANEL_FILES = {"_icml": ["panels_icml.jsonl"], "_other": ["panels_other.jsonl", "panels_other_placebo.jsonl"]}
STALE_REASON = "a cluster row that does not match the current panel files and source parquet"
SOURCE_EXTRA_COLUMNS = {"_icml": [], "_other": ["field", "journal"]}
MARKERS = [
    "commendable",
    "meticulous",
    "intricate",
    "delve",
    "showcas",
    "noteworthy",
    "underscore",
    "pivotal",
    "seamless",
    "invaluable",
    "notably",
    "realm",
    "comprehensive",
    "crucial",
    "robust",
    "leverag",
    "nuanced",
    "multifaceted",
    "paramount",
    "elucidat",
    "garner",
    "tapestry",
    "landscape",
    "holistic",
    "valuable insights",
]
MARKER_RX = re.compile(r"\b(" + "|".join(MARKERS) + ")", re.IGNORECASE)
EDIT_RX = re.compile(
    r"(after|post)[- ]?(the )?(rebuttal|discussion|response)|concerns? (have been|were|was|are) (addressed|resolved)"
    r"|thank(s| you)[^.]{0,40}(response|rebuttal|clarification)|(rais|updat|increas|lower|revis)(e|ed|ing) (my|the) (score|rating|review)"
    r"|\[?(update|revision|edit)d?[:\]]",
    re.IGNORECASE,
)
BATCH_ROWS = 20000
ICLR_FROM = 2020
FIRST_YEAR = {"ICLR": ICLR_FROM}
DEFAULT_FIRST_YEAR = 2019
STD_FORMULA = "hit ~ C(year) + C(size) + bs(log_short, df=4) + bs(log_long, df=4)"
ICLR_SUBMITTED = {2017: 490, 2018: 1018, 2019: 1579, 2020: 2594, 2021: 3014, 2022: 3422, 2023: 4955, 2024: 7404, 2025: 11672}
ICLR_WITHDRAWN = {2017: 0, 2018: 83, 2019: 160, 2020: 369, 2021: 403, 2022: 779, 2023: 1145, 2024: 1661, 2025: 2991}
WITHDRAWALS_OBSERVED_YEAR = 2026
QUINTILE_EDGES = [0, 0.2, 0.4, 0.6, 0.8, 1]


def load_claims(path):
    fields = ("category", "target", "specific")
    claims = {}
    with open(path) as f:
        for row in map(json.loads, f):
            if "claims" in row:
                review_id = row["review_id"]
                claims[review_id] = [{field: claim.get(field) for field in fields} for claim in row["claims"]]
    return claims


def batches(path, columns, ids, desc):
    parquet = pq.ParquetFile(path)
    wanted = pa.array(list(ids), type=parquet.schema_arrow.field("review_id").type)
    n_batches = -(-parquet.metadata.num_rows // BATCH_ROWS)
    for batch in tqdm(parquet.iter_batches(batch_size=BATCH_ROWS, columns=columns), total=n_batches, desc=desc):
        yield batch.filter(pc.is_in(batch.column("review_id"), value_set=wanted))


def review_meta(ids, data=settings.DATA_DIR):
    parts = []
    cols = ["review_id", "paper_id", "conf", "year", "rating", "confidence", "text", "review_time", "modified_time"]
    for batch in batches(data / "reviews.parquet", cols, ids, "review text"):
        chunk = batch.to_pandas()
        chunk["chars"] = chunk.text.str.len()
        chunk["n_markers"] = chunk.text.map(lambda t: len(MARKER_RX.findall(t)))
        chunk["rebuttal_text"] = chunk.text.map(lambda t: bool(EDIT_RX.search(t)))
        parts.append(chunk.drop(columns="text"))
    reviews = pd.concat(parts, ignore_index=True)
    reviews["any_marker"] = reviews.n_markers > 0
    reviews["marker_rate"] = reviews.n_markers / (reviews.chars / 1000)
    ratings = pd.read_parquet(data / "review_cdates.parquet", columns=["review_id", "ext_rating_before", "ext_rating_after"])
    rating_changed = ratings.dropna().query("ext_rating_before != ext_rating_after")
    reviews["edited"] = reviews.review_id.isin(rating_changed.review_id)
    return reviews


def paper_meta(data=settings.DATA_DIR):
    papers = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "conf", "year", "decision", "n_reviews"])
    papers = papers.rename(columns={"n_reviews": "paper_n_reviews"})
    areas = pd.read_parquet(data / "paper_areas.parquet", columns=["paper_id", "area", "area_prob"])
    return papers.merge(areas, on="paper_id", how="left")


def in_main_sample(frame):
    return ~frame.withdrawn & frame.decision.ne("desk_reject")


def review_status(data):
    return pd.read_parquet(data / "reviews.parquet", columns=["review_id", "withdrawn", "decision"])


def excluded_review_ids(data):
    reviews = review_status(data)
    return set(reviews.review_id[~in_main_sample(reviews)])


def main_sample_review_ids(data):
    reviews = review_status(data)
    return set(reviews.review_id[in_main_sample(reviews)])


def review_rows(df, review_ids, paper):
    reviewers = df.groupby("cluster").review.nunique()
    target_reviewers = df.groupby("target").review.nunique()
    df["shared"] = df.cluster.map(reviewers).ge(2)
    df["shared_lenient"] = df.target.map(target_reviewers).ge(2)
    for j, review_id in enumerate(review_ids):
        review_claims = df[df.review == j]
        yield dict(
            review_id=review_id,
            paper_id=paper,
            n_claims=len(review_claims),
            share_shared=review_claims.shared.mean(),
            share_shared_lenient=review_claims.shared_lenient.mean(),
            n_unique=int((~review_claims.shared).sum()),
            share_specific=review_claims.specific.mean(),
            **{f"cat_{c}": float((review_claims.category == c).mean()) for c in CATEGORIES},
        )


def count_true(flags):
    return int(flags.fillna(False).sum())


def claims_of_reviews(df, review_ids):
    if df.empty:
        return df
    position = {review_id: j for j, review_id in enumerate(review_ids)}
    kept = df[df.review_id.isin(review_ids)]
    return kept.assign(review=kept.review_id.map(position))


def placebo_cells(placebo):
    counted = placebo.assign(
        trimmed=placebo.n_excluded_reviews > 0,
        review_pairs=placebo.n_reviews * (placebo.n_reviews - 1) // 2,
    )
    return counted.groupby(["conf", "year", "placebo_kind"], as_index=False).agg(
        panels=("panel_id", "size"), trimmed_panels=("trimmed", "sum"), review_pairs=("review_pairs", "sum")
    )


def ml_panels(model, tag, data, out, reports):
    claims = load_claims(data / f"claims_{model}{tag}.jsonl")
    rmeta = review_meta(claims.keys(), data).set_index("review_id")
    pmeta = paper_meta(data).set_index("paper_id")
    corpus_reviews = set(review_status(data).review_id)
    excluded = set() if tag == WITHDRAWN_TAG else excluded_review_ids(data)

    rows_by_kind = {"panel": [], "placebo": [], "sub3": []}
    dropped = Counter()
    reviews = []
    for panel_id, panel in tqdm(load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id").items(), desc="panels"):
        kind = "placebo" if panel_id.startswith("placebo") else "sub3" if panel_id.startswith("sub3_") else "panel"
        paper = re.sub(r"^(noq|wd|sub3)_", "", panel_id)
        if kind == "panel" and paper not in pmeta.index:
            continue
        if kind == "placebo":
            review_ids = [r for r in panel["review_ids"] if r in corpus_reviews and r not in excluded]
            keep = len(review_ids) >= MIN_PLACEBO_REVIEWS
        else:
            review_ids = panel["review_ids"]
            keep = corpus_reviews.issuperset(review_ids) and excluded.isdisjoint(review_ids)
        if not keep:
            dropped[kind] += 1
            continue
        df = claims_of_reviews(assign(panel, claims), review_ids)
        row = panel_metrics(df, len(review_ids))
        panel_reviews = rmeta.reindex(review_ids)
        row.update(
            panel_id=panel_id,
            total_chars=float(panel_reviews.chars.sum()),
            rating_mean=panel_reviews.rating.mean(),
            rating_sd=panel_reviews.rating.std(ddof=0),
            confidence_mean=panel_reviews.confidence.mean(),
            n_marker_reviews=count_true(panel_reviews.any_marker),
            n_rebuttal_text=count_true(panel_reviews.rebuttal_text),
            n_edited=count_true(panel_reviews.edited),
        )
        if kind == "placebo":
            _, conf, year, _ = panel_id.split("_")
            row.update(
                conf=conf,
                year=int(year),
                placebo_kind="same_area3" if panel_id.startswith("placeboA_") else "shuffled4",
                n_excluded_reviews=len(panel["review_ids"]) - len(review_ids),
            )
        else:
            row.update(pmeta.loc[paper].to_dict(), paper_id=paper)
        rows_by_kind[kind].append(row)
        if kind == "panel" and not df.empty:
            reviews.extend(review_rows(df, review_ids, paper))
    tables = {kind: pd.DataFrame(rows) for kind, rows in rows_by_kind.items() if rows}
    for kind, table in tables.items():
        table.to_parquet(out / f"{kind}_outcomes_{model}{tag}.parquet")
        print(kind, len(table), "panels,", dropped[kind], "dropped:", DROP_REASONS[kind])
    if "placebo" in tables:
        cells = placebo_cells(tables["placebo"])
        cells.to_csv(reports / f"placebo_cells_{model}{tag}.csv", index=False)
        print(cells.to_string(index=False))
    if "panel" in tables:
        review_table = pd.DataFrame(reviews)
        review_table = review_table.merge(rmeta.drop(columns=["paper_id"]), left_on="review_id", right_index=True, how="left")
        review_table = review_table.merge(pmeta[["decision", "area"]], left_on="paper_id", right_index=True, how="left")
        review_table.to_parquet(out / f"review_outcomes_{model}{tag}.parquet")
        print("reviews", len(review_table))


def panel_review_ids(data, names):
    review_ids = {}
    for name in names:
        with open(data / name) as f:
            review_ids.update((panel["panel_id"], panel["review_ids"]) for panel in map(json.loads, f))
    return review_ids


def current_source_panels(data, tag, source_reviews):
    current = panel_review_ids(data, SOURCE_PANEL_FILES[tag])
    return {panel_id: review_ids for panel_id, review_ids in current.items() if source_reviews.issuperset(review_ids)}


def source_panels(model, tag, data, out):
    source = tag[1:]
    extra_columns = SOURCE_EXTRA_COLUMNS[tag]
    claims = load_claims(data / f"claims_{model}{tag}.jsonl")
    meta_columns = ["review_id", "conf", "year", "rating"] + extra_columns
    meta = pd.read_parquet(data / SOURCE_FILES[tag], columns=meta_columns).set_index("review_id")
    current = current_source_panels(data, tag, set(meta.index))
    n_stale = 0
    rows = []
    for panel_id, panel in tqdm(load_jsonl(data / f"clusters_{model}{tag}.jsonl", "panel_id").items(), desc=source):
        if current.get(panel_id) != panel["review_ids"]:
            n_stale += 1
            continue
        row = panel_metrics(assign(panel, claims), len(panel["review_ids"]), ks=(2,))
        first_review = meta.loc[panel["review_ids"][0]]
        row.update(
            panel_id=panel_id,
            placebo=panel_id.startswith("otherplacebo_"),
            conf=first_review.conf,
            year=int(first_review.year),
            rating_sd=meta.rating.reindex(panel["review_ids"]).std(ddof=0),
            **first_review[extra_columns].to_dict(),
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_parquet(out / f"{source}_panel_outcomes_{model}.parquet")
    print(len(rows), "panels,", n_stale, "dropped:", STALE_REASON)
    print(table.groupby(["placebo", "conf", "year"]).hit_rate_strict.mean().mul(100).round(2).unstack(0).to_string())


def panels(
    model=settings.DEFAULT_MODEL,
    tag="",
    data=str(settings.DATA_DIR),
    out=str(settings.DATA_DIR),
    reports=str(settings.REPORTS_DIR),
):
    if tag in SOURCE_FILES:
        source_panels(model, tag, Path(data), Path(out))
    else:
        ml_panels(model, tag, Path(data), Path(out), Path(reports))


def hit(clusters_a, clusters_b):
    same = clusters_a[:, None] == clusters_b[None, :]
    return (same.any(axis=1).mean() + same.any(axis=0).mean()) / 2


def panel_pairs(df, review_ids):
    for a, b in combinations(range(len(review_ids)), 2):
        claims_a = df[df.review == a]
        claims_b = df[df.review == b]
        if claims_a.empty or claims_b.empty:
            continue
        specific_a = claims_a.cluster[claims_a.specific.astype(bool)].to_numpy()
        specific_b = claims_b.cluster[claims_b.specific.astype(bool)].to_numpy()
        yield dict(
            a=review_ids[a],
            b=review_ids[b],
            hit=hit(claims_a.cluster.to_numpy(), claims_b.cluster.to_numpy()),
            hit_spec=hit(specific_a, specific_b) if len(specific_a) and len(specific_b) else np.nan,
        )


def pairs(model=settings.DEFAULT_MODEL, data=str(settings.DATA_DIR), out=str(settings.DATA_DIR)):
    data = Path(data)
    papers = pd.read_parquet(data / "papers.parquet", columns=["paper_id", "conf", "year"]).set_index("paper_id")
    main_reviews = main_sample_review_ids(data)
    n_dropped = n_stale = 0
    rows = []
    for tag in ["", *SOURCE_FILES]:
        clusters_path = data / f"clusters_{model}{tag}.jsonl"
        if not clusters_path.exists():
            continue
        claims = load_claims(data / f"claims_{model}{tag}.jsonl")
        if tag:
            meta = pd.read_parquet(data / SOURCE_FILES[tag], columns=["review_id", "conf", "year"]).set_index("review_id")
            current = current_source_panels(data, tag, set(meta.index))
        for panel_id, panel in tqdm(load_jsonl(clusters_path, "panel_id").items(), desc=f"pairs{tag}"):
            if tag and panel_id.startswith(("icml_", "other_")):
                if current.get(panel_id) != panel["review_ids"]:
                    n_stale += 1
                    continue
                conf, year = meta.loc[panel["review_ids"][0], ["conf", "year"]]
            elif not tag and panel_id in papers.index:
                if not main_reviews.issuperset(panel["review_ids"]):
                    n_dropped += 1
                    continue
                conf, year = papers.loc[panel_id, ["conf", "year"]]
            else:
                continue
            df = assign(panel, claims)
            if df.empty:
                continue
            rows.extend(
                dict(panel_id=panel_id, conf=conf, year=int(year), **pair) for pair in panel_pairs(df, panel["review_ids"])
            )
    table = pd.DataFrame(rows)
    table.to_parquet(Path(out) / f"pairs_{model}.parquet")
    print(n_dropped, "panels dropped:", DROP_REASON)
    print(n_stale, "panels dropped:", STALE_REASON)
    print(table.groupby(["conf", "year"]).agg(pairs=("hit", "size"), hit=("hit", "mean")).to_string())


def review_chars(data, ids):
    parts = []
    for name in ["reviews.parquet", *SOURCE_FILES.values()]:
        for batch in batches(data / name, ["review_id", "text"], ids, name):
            lengths = pc.utf8_length(batch.column("text")).to_numpy()
            parts.append(pd.Series(lengths, index=batch.column("review_id").to_pandas()))
    chars = pd.concat(parts)
    return chars[~chars.index.duplicated()]


def in_window(df):
    return df[df.year >= df.conf.map(FIRST_YEAR).fillna(DEFAULT_FIRST_YEAR)]


def pair_design(model, data, all_years=False):
    design = pd.read_parquet(data / f"pairs_{model}.parquet", columns=["panel_id", "conf", "year", "a", "b", "hit"])
    if not all_years:
        design = in_window(design)
    chars = review_chars(data, set(design.a) | set(design.b))
    log_a = np.log(chars.reindex(design.a).to_numpy())
    log_b = np.log(chars.reindex(design.b).to_numpy())
    design["log_short"], design["log_long"] = np.minimum(log_a, log_b), np.maximum(log_a, log_b)
    sizes = {panel_id: len(review_ids) for panel_id, review_ids in panel_review_ids(data, PANEL_FILES).items()}
    design["size"] = design.panel_id.map(sizes)
    design["size"] = design["size"].clip(upper=design.groupby("conf")["size"].transform("min") + 2)
    return design


def year_rates(S, weights=None):
    import statsmodels.formula.api as smf

    regression = smf.ols(STD_FORMULA, data=S) if weights is None else smf.wls(STD_FORMULA, data=S, weights=weights)
    fit = regression.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(S.panel_id)[0]})
    names = pd.Index(fit.model.exog_names)
    cov = fit.cov_params().to_numpy()
    is_year = names.str.startswith("C(year)")
    x_mean = np.average(fit.model.exog, axis=0, weights=weights)
    rates = {}
    for year in sorted(S.year.unique()):
        x = np.where(is_year, names == f"C(year)[T.{year}]", x_mean)
        rates[year] = (x @ fit.params.to_numpy() * 100, np.sqrt(x @ cov @ x) * 100)
    return rates


def standardize(model=settings.DEFAULT_MODEL, data=str(settings.DATA_DIR), out=None):
    out = out or settings.REPORTS_DIR / f"match_std_{model}.csv"
    design = pair_design(model, Path(data))
    rows = []
    for conf, conf_design in tqdm(design.groupby("conf"), desc="sources"):
        for year, (std, se) in year_rates(conf_design).items():
            year_pairs = conf_design[conf_design.year == year]
            rows.append(dict(conf=conf, year=year, pairs=len(year_pairs), raw=year_pairs.hit.mean() * 100, std=std, std_se=se))
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    print(table.round(2).to_string(index=False))


def rating_icc(reviews):
    z = (reviews.rating - reviews.rating.mean()) / reviews.rating.std()
    return 1 - z.groupby(reviews.paper_id).var().mean() / z.var()


def withdrawals(model=settings.DEFAULT_MODEL, target=0.23, data=str(settings.DATA_DIR), out=None):
    out = out or settings.REPORTS_DIR / f"match_std_withdrawals_{model}.csv"
    data = Path(data)
    reviews = pd.read_parquet(
        data / "reviews.parquet",
        columns=["paper_id", "review_id", "rating", "year", "withdrawn", "decision"],
        filters=[("conf", "==", "ICLR")],
    )
    papers = reviews.groupby("paper_id").agg(
        year=("year", "first"), rating=("rating", "mean"), withdrawn=("withdrawn", "first"), decision=("decision", "first")
    )
    papers["pct"] = papers.groupby("year").rating.rank(pct=True)
    main_sample_rating = papers.rating.where(in_main_sample(papers))
    papers["main_sample_pct"] = main_sample_rating.groupby(papers.year).rank(pct=True)
    decile = np.minimum((papers.pct * 10).astype(int), 9)
    in_observed_year = papers.year == WITHDRAWALS_OBSERVED_YEAR
    withdrawn_by_decile = papers[in_observed_year].groupby(decile[in_observed_year]).withdrawn.mean()
    withdrawn_share = {y: ICLR_WITHDRAWN[y] / ICLR_SUBMITTED[y] for y in ICLR_SUBMITTED}
    papers["drop_share"] = [
        max(0.0, (target - withdrawn_share[y]) / (1 - withdrawn_share[y])) if y in withdrawn_share else 0.0 for y in papers.year
    ]
    papers["withdraw_rate"] = withdrawn_by_decile.reindex(decile).to_numpy()
    year_mean_rate = papers.groupby("year").withdraw_rate.transform("mean")
    papers["weight"] = 1 - np.clip(papers.withdraw_rate * papers.drop_share / year_mean_rate, 0, 1)

    design = pair_design(model, data, all_years=True)
    design = design[design.conf == "ICLR"].copy()
    pair_paper = reviews.set_index("review_id").paper_id.reindex(design.a).to_numpy()
    design["pct"] = papers.pct.reindex(pair_paper).to_numpy()
    design["main_sample_pct"] = papers.main_sample_pct.reindex(pair_paper).to_numpy()
    design["weight"] = papers.weight.reindex(pair_paper).to_numpy()
    design["keep"] = design.pct > papers.drop_share.reindex(pair_paper).to_numpy()
    rates_by_variant = {
        "as published": year_rates(design),
        "weight": year_rates(design, design.weight.to_numpy()),
        "drop": year_rates(design[design.keep]),
    }
    estimates = {variant: {year: rate for year, (rate, _) in rates.items()} for variant, rates in rates_by_variant.items()}
    errors = {f"{variant}_se": {year: se for year, (_, se) in rates.items()} for variant, rates in rates_by_variant.items()}
    table = pd.DataFrame(estimates).join(pd.DataFrame(errors))
    table["withdrawn_share"] = pd.Series(withdrawn_share)
    table["extra_drop"] = papers.groupby("year").drop_share.first()

    rated = reviews[in_main_sample(reviews) & reviews.rating.notna()]
    table["rating_icc"] = rated.groupby("year")[["paper_id", "rating"]].apply(rating_icc)
    panel_hits = (
        design.groupby(["year", "panel_id"]).agg(hit=("hit", "mean"), main_sample_pct=("main_sample_pct", "first")).reset_index()
    )
    quintile = pd.cut(panel_hits.main_sample_pct, QUINTILE_EDGES, labels=[f"q{i}" for i in range(1, 6)])
    hits_by_quintile = panel_hits.groupby(["year", quintile], observed=True).hit.mean().unstack().mul(100)
    table = table.join(hits_by_quintile.add_prefix("raw_rating_"))
    table.index.name = "year"
    table.to_csv(out)
    print(table.round(2).to_string())


if __name__ == "__main__":
    fire.Fire({"panels": panels, "pairs": pairs, "standardize": standardize, "withdrawals": withdrawals})
