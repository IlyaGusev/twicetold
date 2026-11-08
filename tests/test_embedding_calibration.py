import pandas as pd
import pytest

from figures import calibrate_embedding


def test_calibration_uses_fixed_pre_chatgpt_reference_within_venue():
    cells = pd.DataFrame(
        dict(
            conf=["ICLR"] * 4 + ["NeurIPS"] * 2,
            year=[2019, 2020, 2022, 2027, 2021, 2025],
            inter_area=[0.1, 0.4, 0.6, 0.9, 0.2, 0.3],
            intra=[0.2, 0.5, 0.75, 1.0, 0.6, 0.8],
            intra_se=[0.01] * 6,
        )
    )
    out = calibrate_embedding(cells)
    assert out.calibration_baseline.tolist() == pytest.approx([0.5] * 4 + [0.2] * 2)
    assert out.calibrated_intra.tolist() == pytest.approx([-0.6, 0, 0.5, 1, 0.5, 0.75])
    assert out.calibrated_inter_area.iloc[1:3].mean() == pytest.approx(0)
    assert out.calibrated_intra_se.tolist() == pytest.approx([0.02] * 4 + [0.0125] * 2)
    assert "calibrated_intra" not in cells
