import pandas as pd
import pytest

import hub
from jsonl import read_jsonl
from settings import settings

GOLD = settings.GOLD_DIR

pytestmark = pytest.mark.skipif(not (GOLD / "panels_tune.jsonl").exists(), reason="data/gold needs release.py pull")


def test_gold_panels_are_panels_of_the_data():
    kept = list(read_jsonl(GOLD / "panels_tune.jsonl"))
    panels = hub.panel_definitions(["paper", "icml", "journal", "placebo_area", "journal_placebo"])
    assert kept and all(panels[panel["panel_id"]] == panel["review_ids"] for panel in kept)
    reviews = pd.read_parquet(GOLD / "tune_reviews.parquet")
    assert set(reviews.review_id) == {review_id for panel in kept for review_id in panel["review_ids"]}
