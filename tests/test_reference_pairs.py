import numpy as np
import pandas as pd
import pytest

import validate
from settings import settings
from validate import council_pair_label, pair_detection_summary, unique_claim_group_ids


def groups(*members):
    return [{"claim_ids": list(ids), "target_id": i} for i, ids in enumerate(members)]


def test_pair_summary_pools_counts_and_bootstraps_panels():
    panels = pd.DataFrame(dict(pair_tp=[1, 3], pair_fp=[1, 0], pair_fn=[0, 2], pair_candidates=[5, 10]))
    result = pair_detection_summary(panels)
    assert result["pair_precision"] == pytest.approx(80)
    assert result["pair_recall"] == pytest.approx(100 * 4 / 6)
    assert result["pair_f1"] == pytest.approx(100 * 8 / 11)
    assert result["pair_candidates"] == 15
    assert result["pair_accuracy"] == pytest.approx(80)
    assert result["pair_precision_low"] <= 80 <= result["pair_precision_high"]
    assert pair_detection_summary(panels) == result


@pytest.mark.parametrize("label", [True, False])
def test_council_agreement_requires_all_three_labels(label):
    mapping = {"a": 0, "b": 0 if label else 1}
    assert council_pair_label([mapping] * 3, "a", "b") == ("unanimous", label)


def test_council_split_is_not_a_negative_label():
    assert council_pair_label([{"a": 0, "b": 0}, {"a": 0, "b": 1}, {"a": 0, "b": 1}], "a", "b") == ("split", None)


def test_missing_council_member_is_not_a_vote():
    mapping = {"a": 0, "b": 1}
    assert council_pair_label([mapping] * 2, "a", "b") == ("invalid", None)
    assert council_pair_label([mapping, mapping, {}], "a", "b") == ("invalid", None)


def test_repeated_or_missing_claims_cannot_establish_consensus():
    mapping = unique_claim_group_ids(groups(("a", "b"), ("a", "c")))
    assert mapping == {"b": 0, "c": 1}
    assert council_pair_label([mapping] * 3, "a", "b") == ("invalid", None)
    assert council_pair_label([mapping] * 3, "missing", "b") == ("invalid", None)


def test_council_command_uses_only_raw_group_votes(monkeypatch, tmp_path):
    panel = dict(panel_id="P", review_ids=["a", "b"], kind="same", conf="ICLR", year=2024, stratum="plain", split="dev")
    claims = {rid: [{"claim": "problem"}] for rid in panel["review_ids"]}
    raw = [dict(panel, model=model, clusters=groups(("R0-0", "R1-0"))) for model in settings.GOLD_COUNCIL]
    ours = {"P": dict(panel, clusters=groups(("R0-0",), ("R1-0",)))}
    monkeypatch.setattr(validate, "read_jsonl", lambda path: [panel] if path.name == "panels_tune.jsonl" else raw)
    monkeypatch.setattr(validate.extract, "load_claims", lambda *paths: claims)
    monkeypatch.setattr(validate, "load_jsonl", lambda path, key: ours)
    validate.council_agreement(gold=tmp_path, reports=tmp_path, tex=tmp_path / "council.tex")
    pairs = pd.read_csv(tmp_path / f"council_agreement_pairs_{settings.DEFAULT_MODEL}.csv")
    assert pairs[settings.GOLD_COUNCIL].all().all()
    assert pairs.status.tolist() == ["unanimous"]
    assert not pairs.predicted.any()
    summary = pd.read_csv(tmp_path / f"council_agreement_{settings.DEFAULT_MODEL}.csv")
    assert (summary.pair_fn == 1).all()
    assert (summary.pair_recall == 0).all()


def test_no_positive_pairs_leave_recall_undefined():
    panels = pd.DataFrame(dict(pair_tp=[0, 0], pair_fp=[1, 1], pair_fn=[0, 0], pair_candidates=[2, 2]))
    result = pair_detection_summary(panels)
    assert result["pair_precision"] == 0
    assert np.isnan(result["pair_recall"])
    assert np.isnan(result["pair_recall_low"])
