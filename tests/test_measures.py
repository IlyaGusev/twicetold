import pandas as pd
import pytest

from measures import pair_measures, third_review_novelty, unique_counts


@pytest.mark.parametrize(
    ("smaller", "larger", "hit", "jaccard", "dice"),
    [(10, 10, 20.0, 100 * 2 / 18, 20.0), (4, 10, 50.0, 100 * 2 / 12, 100 * 4 / 14)],
)
def test_two_shared_problems(smaller, larger, hit, jaccard, dice):
    pairs = pd.DataFrame(dict(hit=[2 / smaller], shared=[2], smaller=[smaller], larger=[larger], union=[smaller + larger - 2]))
    row = pair_measures(pairs).iloc[0]
    assert (row.hit, row.jaccard, row.dice) == pytest.approx((hit, jaccard, dice))


def test_empty_pair_scores_zero():
    row = pair_measures(pd.DataFrame(dict(hit=[0.0], shared=[0], smaller=[0], larger=[0], union=[0]))).iloc[0]
    assert (row.jaccard, row.dice) == (0.0, 0.0)


def test_unique_counts():
    assert unique_counts([{1, 2, 3}, {2, 4}, {3, 5}]) == [1, 1, 1]


def test_third_review_novelty_averages_trios_and_members():
    assert third_review_novelty([{1, 2}, {1}, {3}]) == pytest.approx((1 + 0 + 1) / 3)
    groups = [{1}, {1}, {1}, {2}]
    trios = [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    assert third_review_novelty(groups) == pytest.approx(sum(map(sum, trios)) / 12)
