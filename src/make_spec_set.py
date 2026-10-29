"""Sample 300 claims (150 from ICLR 2017-2022, 150 from 2024-2026) for the generic-vs-specific validation.
Writes data/spec/spec_input.jsonl (claim text only) and data/spec/spec_key.parquet (pipeline labels)."""

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
T = pd.read_parquet(DATA / "claims_table_gpt-6-luna.parquet", columns=["review_id", "k", "conf", "year", "claim", "specific"])
T = T[T.conf == "ICLR"]
pre = T[T.year.between(2017, 2022)].sample(150, random_state=11)
post = T[T.year.between(2024, 2026)].sample(150, random_state=11)
S = pd.concat([pre, post]).sample(frac=1, random_state=11).reset_index(drop=True)
S["item_id"] = [f"C{i:03d}" for i in range(len(S))]
S.to_parquet(DATA / "spec" / "spec_key.parquet")
with open(DATA / "spec" / "spec_input.jsonl", "w") as f:
    for r in S.itertuples():
        f.write(json.dumps({"item_id": r.item_id, "claim": r.claim}) + "\n")
print(len(S), "claims; pipeline specific share", round(S.specific.mean(), 3))
