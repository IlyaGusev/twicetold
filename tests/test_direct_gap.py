import numpy as np
import pandas as pd

from validate import direct_gap, paper_gap


def design():
    return pd.DataFrame(
        dict(
            panel_id=["p", "p", "p", "q", "q", "r"],
            a=["p1", "p1", "p3", "q1", "q3", "r1"],
            b=["p2", "p2", "p4", "q2", "q4", "r2"],
            hit=[1.0, 0.5, 0.0, 1.0, 0.0, 1.0],
        )
    )


FLAG = pd.Series(dict(p1=1, p2=1, p3=0, p4=0, q1=1, q2=1, q3=0, q4=0, r1=1, r2=0))


def test_direct_gap_is_the_mean_paper_difference():
    gap = direct_gap(design(), FLAG)
    assert gap["two"] == 87.5
    assert gap["two_se"] == 12.5
    assert gap["papers"] == 2


def test_no_paper_with_both_pair_types_gives_no_estimate():
    one_flagged = design().query("panel_id == 'r'")
    gap = direct_gap(one_flagged, FLAG)
    assert np.isnan(gap["two"])
    assert gap["papers"] == 0


def test_paper_gap_aligns_groups_by_position():
    groups = pd.Series([2, 2, 0, 2, 0, 1], index=list("uvwxyz"))
    assert paper_gap(design(), groups, 2, 0) == dict(est=87.5, se=12.5, papers=2)
