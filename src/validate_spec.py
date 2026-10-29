"""Generic-vs-specific validation: agreement of two independent annotators with each other and with the pipeline
label, overall and by period. Usage: validate_spec.py -> reports/spec_validation.txt"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DATA = Path(__file__).resolve().parent.parent / "data" / "spec"
K = pd.read_parquet(DATA / "spec_key.parquet").set_index("item_id")
for a in "ab":
    J = {r["item_id"]: bool(r["specific"]) for r in map(json.loads, open(DATA / f"judgments_{a}.jsonl"))}
    K[f"ann_{a}"] = K.index.map(J)
K["pl"] = K.specific.astype(bool)
K["period"] = np.where(K.year <= 2022, "2017-2022", "2024-2026")
K["gold"] = K.ann_a & K.ann_b  # both annotators say specific


def block(d, name):
    print(f"== {name} (n={len(d)})")
    print(f"  share specific: pipeline {d.pl.mean():.3f}, annotator A {d.ann_a.mean():.3f}, annotator B {d.ann_b.mean():.3f}")
    print(f"  kappa A vs B {cohen_kappa_score(d.ann_a, d.ann_b):.3f}; agreement {np.mean(d.ann_a == d.ann_b):.3f}")
    for a in "ab":
        print(f"  kappa pipeline vs {a.upper()} {cohen_kappa_score(d.pl, d[f"ann_{a}"]):.3f}; agreement {np.mean(d.pl == d[f'ann_{a}']):.3f}")
    both = d[d.ann_a == d.ann_b]
    print(f"  on the {len(both)} items where A and B agree: pipeline accuracy {np.mean(both.pl == both.ann_a):.3f}, "
          f"precision {np.mean(both[both.pl].ann_a):.3f}, recall {np.mean(both[both.ann_a].pl):.3f}")


block(K, "all")
for p, d in K.groupby("period"):
    block(d, p)
