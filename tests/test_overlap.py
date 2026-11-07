import pandas as pd
import pytest

from figures import summarize
from metrics import panel_metrics
from outcomes import hit, panel_pairs


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


def test_empty_pair_scores_zero():
    assert hit([], [1]) == 0.0


def panel_frame(groups):
    rows = [
        dict(review=review, cluster=group, target=group, specific=True, category="novelty")
        for review, problems in enumerate(groups)
        for group in problems
    ]
    return pd.DataFrame(rows)


def test_each_paper_has_equal_weight():
    shared = [[1, 2]] * 5
    disjoint = [[1], [2]]
    panels = pd.DataFrame(
        [dict(conf="ICLR", year=2026, **panel_metrics(panel_frame(groups), len(groups))) for groups in (shared, disjoint)]
    )
    row = summarize(panels, panels.iloc[:0]).loc[("ICLR", 2026)]
    assert row.match == pytest.approx(50.0)
    assert row.match_se == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ([range(10), range(20)], 1.0),
        ([range(10), range(20), []], 1 / 3),
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
