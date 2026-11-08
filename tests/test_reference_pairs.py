import numpy as np
import pandas as pd
import pytest

import validate
from settings import settings
from validate import aligned_claim_ids, council_pair_label, pair_detection_summary, unique_claim_group_ids


def groups(*members):
    return [{"claim_ids": list(ids), "target_id": i} for i, ids in enumerate(members)]


def test_alignment_keeps_one_to_one_claim_ids():
    ids = ["a", "b"]
    claims = {rid: [{"claim": str(i)} for i in range(3)] for rid in ids}
    matches = {rid: {"clusters": groups(*[(f"R0-{i}", f"R1-{i}") for i in range(3)])} for rid in ids}
    model = {"review_ids": ids}
    assert aligned_claim_ids(ids, claims, claims, model, matches) == [
        (rid, f"R{j}-{i}", f"R{j}-{i}") for j, rid in enumerate(ids) for i in range(3)
    ]


def test_alignment_excludes_splits_and_failed_matches_and_handles_review_order():
    ids = ["a", "b", "c"]
    claims = {rid: [{"claim": str(i)} for i in range(2)] for rid in ids}
    matches = {
        "a": {"clusters": groups(("R0-0", "R1-0"), ("R0-1", "R1-1"))},
        "b": {"clusters": groups(("R0-0", "R1-0"), ("R0-1", "R1-1", "R1-2"))},
    }
    model_claims = claims | {"b": [{"claim": str(i)} for i in range(3)]}
    model = {"review_ids": ids[::-1]}
    assert aligned_claim_ids(ids, claims, model_claims, model, matches) == [
        ("a", "R0-0", "R2-0"),
        ("a", "R0-1", "R2-1"),
        ("b", "R1-0", "R1-0"),
    ]


def test_empty_reviews_have_no_aligned_pairs():
    claims = {"a": [], "b": []}
    panel = {"review_ids": ["a", "b"]}
    matches = {rid: {"clusters": []} for rid in claims}
    assert aligned_claim_ids(panel["review_ids"], claims, claims, panel, matches) == []


def test_pair_summary_pools_counts_and_bootstraps_panels():
    panels = pd.DataFrame(dict(pair_tp=[1, 3], pair_fp=[1, 0], pair_fn=[0, 2], pair_candidates=[5, 10], pair_aligned=[4, 6]))
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
    panel = dict(panel_id="P", review_ids=["a", "b"], kind="same", split="dev", year=2024)
    claims = {rid: {"claims": [{"claim": "problem"}]} for rid in panel["review_ids"]}
    matches = {rid: {"clusters": groups(("R0-0", "R1-0"))} for rid in panel["review_ids"]}
    raw = [dict(panel, model=model, clusters=groups(("R0-0", "R1-0"))) for model in settings.GOLD_COUNCIL]
    model_panel = dict(panel, clusters=groups(("R0-0",), ("R1-0",)))
    monkeypatch.setattr(validate.hub, "root", lambda: tmp_path)
    monkeypatch.setattr(validate.hub, "clusters", lambda *args, **kwargs: {"P": model_panel})
    monkeypatch.setattr(validate.hub, "panel_claims", lambda *args: {rid: row["claims"] for rid, row in claims.items()})
    monkeypatch.setattr(validate, "read_jsonl", lambda path: [panel] if path.name == "panels_tune.jsonl" else raw)

    def load(path, key):
        assert path.parent.name == "adjudicate" or path.name == f"clusters_{settings.GOLD_CLAIM_JUDGE_MODEL}.jsonl"
        return claims if path.parent.name == "adjudicate" else matches

    monkeypatch.setattr(validate, "load_jsonl", load)
    validate.council_agreement(reports=tmp_path)
    pairs = pd.read_csv(tmp_path / f"council_agreement_pairs_{settings.DEFAULT_MODEL}.csv")
    assert "judge" not in pairs
    assert pairs[settings.GOLD_COUNCIL].all().all()
    assert pairs.status.tolist() == ["unanimous"]
    assert not pairs.luna.any()
    summary = pd.read_csv(tmp_path / f"council_agreement_{settings.DEFAULT_MODEL}.csv")
    assert set(summary.reference) == {"unanimous"}
    assert (summary.pair_fn == 1).all()


def test_no_positive_pairs_leave_recall_undefined():
    panels = pd.DataFrame(dict(pair_tp=[0, 0], pair_fp=[1, 1], pair_fn=[0, 0], pair_candidates=[2, 2], pair_aligned=[2, 2]))
    result = pair_detection_summary(panels)
    assert result["pair_precision"] == 0
    assert np.isnan(result["pair_recall"])
    assert np.isnan(result["pair_recall_low"])
