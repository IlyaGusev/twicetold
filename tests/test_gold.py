import pandas as pd
import pytest

import gold
from jsonl import read_jsonl
from settings import settings

GOLD = settings.GOLD_DIR
LFS_POINTER = b"version https://git-lfs"

pytestmark = pytest.mark.skipif(
    not (GOLD / "panels_tune.jsonl").exists() or (GOLD / "panels_tune.jsonl").read_bytes().startswith(LFS_POINTER),
    reason="data/gold needs git lfs pull",
)


def test_deterministic_steps_rebuild_data_gold(tmp_path):
    gold.sample(tmp_path)
    sampled = {panel["panel_id"]: panel for panel in read_jsonl(tmp_path / "panels_tune.jsonl")}
    kept = list(read_jsonl(GOLD / "panels_tune.jsonl"))
    assert kept and all(sampled[panel["panel_id"]] == panel for panel in kept)
    rebuilt = pd.read_parquet(tmp_path / "tune_reviews.parquet")
    assert rebuilt.reset_index(drop=True).equals(pd.read_parquet(GOLD / "tune_reviews.parquet").reset_index(drop=True))
