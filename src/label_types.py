"""Short readable labels for the k-means critique types. Usage: label_types.py <model>"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

DATA = Path(__file__).resolve().parent.parent / "data"
REPORTS = Path(__file__).resolve().parent.parent / "reports"
rng = np.random.default_rng(0)
client = OpenAI(max_retries=4)
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label"], "properties": {"label": {"type": "string"}}}


def main(model):
    T = pd.read_parquet(DATA / f"claims_table_{model}.parquet", columns=["conf", "year", "claim", "ctype"])
    pre = T[(T.conf == "ICLR") & T.year.between(2017, 2022)]
    types = pd.read_csv(REPORTS / f"critique_types_{model}.csv", index_col=0)

    def one(c):
        ex = pre[pre.ctype == c].claim
        ex = ex.sample(min(15, len(ex)), random_state=0).tolist()
        r = client.responses.create(model="gpt-6-luna", instructions="These are peer-review critique claims that fall in one cluster. Give the cluster a label of at most 8 words naming the shared criticism (e.g. 'missing comparison with simple baselines', 'theorem assumptions unrealistic'). Return JSON.",
                                    input="\n".join(f"- {e}" for e in ex), reasoning={"effort": "low"},
                                    text={"format": {"type": "json_schema", "name": "label", "schema": SCHEMA, "strict": True}})
        return c, json.loads(r.output_text)["label"]

    with ThreadPoolExecutor(16) as pool:
        labels = dict(pool.map(one, types.index))
    types["short"] = pd.Series(labels)
    types.to_csv(REPORTS / f"critique_types_{model}.csv")
    sh = pd.read_csv(REPORTS / f"type_shift_{model}.csv", index_col=0)
    sh["short"] = sh.index.map(labels)
    sh.to_csv(REPORTS / f"type_shift_{model}.csv")
    print(sh[["short", "f2023", "f2026", "delta"]].head(15).to_string())
    print(sh[["short", "f2023", "f2026", "delta"]].tail(15).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
