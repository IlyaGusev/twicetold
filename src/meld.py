import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from ai import TAGS
from jsonl import read_jsonl
from settings import settings

INTEL_AI_PER_LLM = 1000
FLUSH_DOCS = 2000
REAL_SETS = ["sample", "icml", "journal"]
KNOWN_SETS = {
    "human": "Human, pre-2023",
    "heldout": "Human, ICLR 2023",
    "polish": "Polished",
    "rewrite": "From outline",
    "rewrite_panel": "Panels: outline, Luna",
    "polish_panel": "Panels: polished, Luna",
    "rewrite_panel_second": "Panels: outline, Kimi",
    "polish_panel_second": "Panels: polished, Kimi",
    "intel_ai": "From paper, five LLMs",
    "paper_review": "From paper, Luna",
    "paper_review_second": "From paper, Kimi",
}


GENERATED_SETS = [
    ("rewrite_panel", settings.REWRITE_TEST_DIR / f"rewrites_{settings.AI_REFERENCE_MODEL}.jsonl"),
    ("polish_panel", settings.REWRITE_TEST_DIR / f"polishes_{settings.AI_REFERENCE_MODEL}.jsonl"),
    ("rewrite_panel_second", settings.REWRITE_TEST_DIR / f"rewrites_{settings.REWRITE_SECOND_WRITER}.jsonl"),
    ("polish_panel_second", settings.REWRITE_TEST_DIR / f"polishes_{settings.REWRITE_SECOND_WRITER}.jsonl"),
    ("paper_review", settings.PAPER_REVIEW_DIR / f"reviews_{settings.AI_REFERENCE_MODEL}.jsonl"),
    ("paper_review_second", settings.PAPER_REVIEW_DIR / f"reviews_{settings.REWRITE_SECOND_WRITER}.jsonl"),
]


def labels(known, human, heldout, reports, tex):
    df = pd.concat([human, heldout, known])
    df = df[df.known.isin(KNOWN_SETS)]
    out = df.groupby("known").agg(
        texts=("flag", "size"),
        unflagged=("flag", lambda f: 100 * (1 - f.mean())),
        written=("written", lambda w: 100 * w.mean()),
    )
    out["edited"] = 100 - out.unflagged - out.written
    out = out.loc[[k for k in KNOWN_SETS if k in out.index]]
    out.round(2).to_csv(Path(reports) / "meld_labels.csv")
    print(out.round(1).to_string())
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Texts & $n$ & Unflagged & Edited & Written \\",
        r"\midrule",
    ]
    lines += [
        f"{KNOWN_SETS[k]} & {int(r.texts):,} & {r.unflagged:.1f} & {r.edited:.1f} & {r.written:.1f} \\\\"
        for k, r in out.iterrows()
    ]
    Path(tex).write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def build_input(out=settings.RAW_DIR / "meld_input.parquet"):
    cols = ["review_id", "conf", "year", "text"]
    sample = pd.read_parquet(settings.RAW_DIR / "sample_reviews.parquet", columns=cols).assign(set="sample")
    icml = pd.read_parquet(settings.RAW_DIR / "icml_sample.parquet", columns=cols).assign(set="icml")
    journal = pd.read_parquet(settings.RAW_DIR / "other_reviews.parquet", columns=cols).assign(set="journal")
    intel = pd.read_parquet(
        settings.ADDITIONAL_DIR / "intel_reviews.parquet", columns=["review_id", "llm", "year", "text", "label"]
    )
    intel = intel[intel.label == 1].groupby("llm").sample(INTEL_AI_PER_LLM, random_state=0)
    intel = intel.assign(review_id=intel.llm + ":" + intel.review_id, conf="ICLR", set="intel_ai")[cols + ["set"]]
    rewrites = pd.read_json(settings.ADDITIONAL_DIR / f"ai_reference_{settings.AI_REFERENCE_MODEL}.jsonl", lines=True)
    rewrites = rewrites[~rewrites.conf.isin(["ICLR", "NeurIPS"])].assign(set="rewrite")[cols + ["set"]]
    polish = pd.read_json(settings.ADDITIONAL_DIR / f"ai_polish_{settings.AI_REFERENCE_MODEL}.jsonl", lines=True)
    polish = polish.assign(set="polish")[cols + ["set"]]
    generated = [pd.read_json(path, lines=True).assign(set=name)[cols + ["set"]] for name, path in GENERATED_SETS]
    df = pd.concat([sample, icml, journal, intel, rewrites, polish, *generated], ignore_index=True)
    df["key"] = df.set + ":" + df.review_id
    assert df.key.is_unique
    print(df.groupby("set").size().to_string())
    df.to_parquet(out)


def load_scorer():
    model_dir = snapshot_download(settings.MELD_REPO, revision=settings.MELD_REVISION)
    spec = importlib.util.spec_from_file_location("meld_remote", Path(model_dir) / "meld.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.Scorer(model_dir, device="cuda")


def windows(module, scorer, text):
    ids = scorer.tok(module.canonicalize(text), add_special_tokens=False, truncation=False)["input_ids"][: scorer.cap]
    return [ids[i : i + scorer.window] for i in range(0, len(ids), scorer.window)]


def score_docs(module, scorer, texts, batch_tokens):
    parts = []
    for doc, text in enumerate(texts):
        parts += [(doc, w, ids) for w, ids in enumerate(windows(module, scorer, text))]
    parts.sort(key=lambda p: len(p[2]))
    outputs = defaultdict(dict)
    start = 0
    while start < len(parts):
        end = start + 1
        while end < len(parts) and (end - start + 1) * (len(parts[end][2]) + 2) <= batch_tokens:
            end += 1
        batch = parts[start:end]
        width = max(len(p[2]) for p in batch) + 2
        inp = torch.full((len(batch), width), scorer.pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for j, (_, _, ids) in enumerate(batch):
            inp[j, 0] = scorer.cls_id
            inp[j, 1 : 1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
            inp[j, 1 + len(ids)] = scorer.sep_id
            att[j, : len(ids) + 2] = 1
        s_tok, c_tok, o_tok = scorer.model.token_scores(inp.to(scorer.device), att.to(scorer.device))
        for j, (doc, w, ids) in enumerate(batch):
            span = slice(1, 1 + len(ids))
            outputs[doc][w] = (c_tok[j, span].cpu(), s_tok[j, span].cpu(), o_tok[j, span].cpu())
        start = end
    results = []
    for doc in range(len(texts)):
        chunks = [outputs[doc][w] for w in sorted(outputs[doc])]
        if not chunks:
            results.append(dict(score=None, n_tokens=0, family=None, task=None))
            continue
        c_tok, s_tok, o_tok = (torch.cat([c[i] for c in chunks]) for i in range(3))
        results.append(
            dict(
                score=float(module.top_quantile_mean(c_tok, scorer.rho)),
                n_tokens=int(c_tok.numel()),
                family=scorer.cfg["families"][int(module.top_quantile_mean(s_tok, scorer.rho).argmax())],
                task=scorer.cfg["ops"][int(module.top_quantile_mean(o_tok, scorer.rho).argmax())],
            )
        )
    return results


def check(input=settings.RAW_DIR / "meld_input.parquet", n=64, batch_tokens=16384):
    module, scorer = load_scorer()
    texts = pd.read_parquet(input, columns=["text"]).sample(n, random_state=0).text.str.replace(TAGS, " ", regex=True).tolist()
    batched = [r["score"] for r in score_docs(module, scorer, texts, batch_tokens)]
    single = [scorer.score(t)["score"] for t in texts]
    diff = max(abs(a - b) for a, b in zip(batched, single))
    print(f"max |batched - single| over {n} texts: {diff:.2e}")


def scores(input=settings.RAW_DIR / "meld_input.parquet", out=settings.RAW_DIR / "meld_scores.jsonl", batch_tokens=16384):
    out = Path(out)
    df = pd.read_parquet(input, columns=["key", "text"])
    done = {row["key"] for row in read_jsonl(out)} if out.exists() else set()
    df = df[~df.key.isin(done)]
    df = df.assign(text=df.text.str.replace(TAGS, " ", regex=True), n=df.text.str.len()).sort_values("n")
    print(f"{len(done)} done, {len(df)} to score -> {out}", flush=True)
    module, scorer = load_scorer()
    with open(out, "a") as f:
        for start in range(0, len(df), FLUSH_DOCS):
            chunk = df.iloc[start : start + FLUSH_DOCS]
            for key, result in zip(chunk.key, score_docs(module, scorer, chunk.text.tolist(), batch_tokens)):
                f.write(json.dumps(dict(key=key, **result)) + "\n")
            f.flush()
            print(f"{start + len(chunk)}/{len(df)}", flush=True)


def flags(
    input=settings.RAW_DIR / "meld_input.parquet",
    scored=settings.RAW_DIR / "meld_scores.jsonl",
    out=settings.RAW_DIR / "meld_flags.parquet",
    reports=settings.REPORTS_DIR,
    tex=settings.TABLES_DIR / "meld_labels.tex",
):
    df = pd.read_parquet(input, columns=["key", "set", "review_id", "conf", "year"]).merge(
        pd.read_json(scored, lines=True)[["key", "score", "n_tokens", "family", "task"]], on="key"
    )
    is_real = df.set.isin(REAL_SETS)
    calibration = df[is_real & (df.year <= settings.MELD_CALIBRATION_LAST_YEAR)]
    threshold = calibration.score.quantile(settings.MELD_FLAG_QUANTILE)
    df["flag"] = df.score > threshold
    df["written"] = df.flag & (df.task == settings.MELD_WRITTEN_TASK)
    rates = df.groupby(["set", "conf", "year"]).agg(n=("flag", "size"), flagged=("flag", "mean"), written=("written", "mean"))
    rates[["flagged", "written"]] *= 100
    rates.round(3).to_csv(Path(reports) / "meld_rates.csv")
    print(rates.round(2).to_string())
    pre = calibration.assign(flag=calibration.score > threshold)
    pre_by_source = pre.groupby("conf").flag.mean() * 100
    pre_by_year = pre.groupby(["conf", "year"]).flag.mean() * 100
    known = rates.loc[["intel_ai", "rewrite"]]
    summary = {
        "threshold": threshold,
        "calibration_reviews": len(calibration),
        "pre_source_max": pre_by_source.max(),
        "pre_year_max": pre_by_year.max(),
        "intel_ai_reviews": int(known.loc["intel_ai"].n.sum()),
        "intel_ai_recall": known.loc["intel_ai"].flagged.min(),
        "rewrite_reviews": int(known.loc["rewrite"].n.sum()),
        "rewrite_recall_min": known.loc["rewrite"].flagged.min(),
    }
    heldout = df[is_real & (df.conf == "ICLR") & (df.year == settings.MELD_CALIBRATION_LAST_YEAR + 1)]
    labels(
        df[~is_real].assign(known=df.set),
        calibration.assign(known="human", flag=pre.flag, written=pre.flag & (pre.task == settings.MELD_WRITTEN_TASK)),
        heldout.assign(known="heldout"),
        reports,
        tex,
    )
    pd.Series(summary).to_csv(Path(reports) / "meld_summary.csv", header=["value"], index_label="name")
    print(pd.Series(summary).round(3).to_string())
    real = df[is_real]
    assert real.review_id.is_unique
    real[["review_id", "conf", "year", "score", "n_tokens", "family", "task", "flag", "written"]].to_parquet(out, index=False)


if __name__ == "__main__":
    fire.Fire({"input": build_input, "check": check, "scores": scores, "flags": flags})
