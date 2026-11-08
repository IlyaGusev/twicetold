import pandas as pd
import pytest

import gold
from settings import settings

GOLD = settings.GOLD_DIR
LFS_POINTER = b"version https://git-lfs"

pytestmark = pytest.mark.skipif(
    not (GOLD / "panels_tune.jsonl").exists() or (GOLD / "panels_tune.jsonl").read_bytes().startswith(LFS_POINTER),
    reason="data/gold needs git lfs pull",
)


def same_files(out, pattern):
    names = sorted(p.relative_to(GOLD) for p in GOLD.glob(pattern))
    assert names
    for name in names:
        assert (out / name).read_bytes() == (GOLD / name).read_bytes(), name


def test_deterministic_steps_rebuild_data_gold(tmp_path):
    gold.sample(tmp_path)
    same_files(tmp_path, "panels_tune.jsonl")
    rebuilt = pd.read_parquet(tmp_path / "tune_reviews.parquet")
    assert rebuilt.reset_index(drop=True).equals(pd.read_parquet(GOLD / "tune_reviews.parquet").reset_index(drop=True))
