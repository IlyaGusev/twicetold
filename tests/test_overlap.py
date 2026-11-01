import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from metrics import panel_metrics
from outcomes import hit, panel_pairs, year_rates
from validate import audit_overlap


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (list(range(10)), list(range(20)), 1.0),
        (list(range(20)), list(range(10)), 1.0),
        ([1, 2, 3], [2, 3, 4, 5], 2 / 3),
        ([1, 2], [3, 4], 0.0),
        ([1, 1, 2], [1, 3, 4], 0.5),
        ([1, 1, 1, 2], [1, 3], 0.5),
        ([1], [1, 2, 3], 1.0),
    ],
)
def test_overlap(a, b, expected):
    assert hit(a, b) == pytest.approx(expected)


def test_empty_pair_is_undefined():
    assert math.isnan(hit([], [1]))


def test_audit_merges_many_to_one_matches():
    row = dict(
        a_claims=["same", "same", "unique"],
        b_claims=["same", "other", "another"],
        matches=[["A1", "B1"], ["A2", "B1"]],
    )
    assert audit_overlap(row) == 0.5


def test_each_paper_has_equal_weight():
    pairs = pd.DataFrame([dict(year=2026, panel_id="A", hit=1.0)] * 10 + [dict(year=2026, panel_id="B", hit=0.0)])
    mean, error = year_rates(pairs)[2026]
    assert mean == 50.0
    assert error == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ([range(10), range(20)], 1.0),
        ([range(10), range(20), []], 1.0),
        ([[1], [1, 2], [2]], 2 / 3),
        ([[1, 1, 2], [1, 3]], 0.5),
    ],
)
def test_panel_uses_same_measure(groups, expected):
    rows = [
        dict(review=review, cluster=group, target=group, specific=True, category="novelty")
        for review, problems in enumerate(groups)
        for group in problems
    ]
    frame = pd.DataFrame(rows)
    result = panel_metrics(frame, len(groups))
    pairs = list(panel_pairs(frame, list(range(len(groups)))))
    assert result["hit_rate_strict"] == pytest.approx(expected)
    assert result["hit_rate_lenient"] == pytest.approx(expected)
    assert sum(pair["hit"] for pair in pairs) / len(pairs) == pytest.approx(expected)
