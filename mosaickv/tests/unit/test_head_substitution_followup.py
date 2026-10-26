"""Synthetic numerical/null-control tests; never benchmark-quality evidence."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_head_substitution import fixture, row

from mosaickv.backends.hf_allocation_guard import draft_row_causes, packed_positions_sha
from mosaickv.backends.hf_head_substitution import shadow_logit_metrics
from mosaickv.backends.hf_runtime import PrefillDiagnosticComplete
from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.evaluation.head_substitution_diagnostic import HeadDiagnosticPolicy
from mosaickv.evaluation.head_substitution_followup import (
    HeadSubstitutionFollowup,
    audit_null_control,
    clone_anchor,
    repack_anchor,
)
from mosaickv.measurements.cache_lifecycle import CacheLifecycle
from mosaickv.selection.draft_shortlist import draft_shortlist
from mosaickv.selection.head_substitution import pack_substitution, substitution_pool


def test_shadow_fp64_is_additional_and_does_not_replace_the_registered_guard():
    reference = torch.tensor([[1.0, 0.0]], dtype=torch.float16)
    candidate = torch.tensor([[0.0, 1.0]], dtype=torch.float16)
    assert shadow_logit_metrics(reference, reference)["logit_kl_fp64"] == 0
    observed = shadow_logit_metrics(reference, candidate)
    expected = (torch.softmax(reference.double(), -1)[0] * torch.tensor([1, -1])).sum()
    assert observed["logit_kl_fp64"] == pytest.approx(expected.item(), abs=1e-15)
    assert observed["model_logits_dtype"] == "torch.float16"
    assert observed["reference_greedy_token_ids"] == [0]
    assert observed["candidate_greedy_token_ids"] == [1]
    metrics = row(1, 0.2)
    metrics["candidate"]["logit_kl_fp64"] = 0
    metrics["anchor"]["logit_kl_fp64"] = 1
    assert draft_row_causes(metrics, DraftAllocationGuardConfig()) == ["logit_kl"]
    for invalid in (torch.tensor([[float("nan"), 0]]), torch.zeros(1, 3)):
        with pytest.raises(ValueError, match="matching finite"):
            shadow_logit_metrics(reference, invalid)


def test_null_clone_and_repack_keep_exact_tensor_slots_and_independent_storage():
    full, anchor, _ = fixture()
    for control in (clone_anchor(anchor), repack_anchor(full, anchor)):
        assert packed_positions_sha(control) == packed_positions_sha(anchor)
        assert control.slot_records == anchor.slot_records
        for old, new in zip(anchor.snapshot.layers, control.snapshot.layers, strict=True):
            assert torch.equal(old.key, new.key) and torch.equal(old.value, new.value)
            assert old.key.data_ptr() != new.key.data_ptr()
            assert old.value.data_ptr() != new.value.data_ptr()
        assert CacheLifecycle.count(
            [(layer.key, layer.value) for layer in anchor.snapshot.layers]
        ) == (CacheLifecycle.count([(layer.key, layer.value) for layer in control.snapshot.layers]))


def null_rows():
    rows = []
    for step in range(1, 33):
        r = row(step)
        r["anchor"]["active_bytes"] = 512
        r["candidate"] = dict(
            r["anchor"],
            bitwise_equal_to_anchor_logits=True,
            anchor_candidate_maximum_logit_difference=0.0,
            anchor_candidate_kl_fp64=0.0,
        )
        rows.append(r)
    return rows


def test_null_controls_require_all_32_prefixes_and_bitwise_not_merely_guard_identity():
    rows = null_rows()
    assert audit_null_control(rows, DraftAllocationGuardConfig())["passed"]
    rows[-1]["candidate"]["bitwise_equal_to_anchor_logits"] = False
    assert not audit_null_control(rows, DraftAllocationGuardConfig())["passed"]
    assert audit_null_control(rows, DraftAllocationGuardConfig())["registered_guard"]["accepted"]
    with pytest.raises(ValueError, match="entire registered horizon"):
        audit_null_control(rows[:4], DraftAllocationGuardConfig())


def test_extracts_all_and_only_three_unextended_survivors_without_new_proposals():
    from prepare_head_followup import extract_survivors

    def probe(_):
        for step in range(1, 33):
            yield row(step, 0.2 if step == 5 else 0.05)

    selection = draft_shortlist(
        tuple((str(i), i) for i in range(5)), probe, DraftAllocationGuardConfig()
    )
    result = dict(
        selection=selection,
        anchor_positions_sha256="a" * 64,
        proposals={str(i): {"synthetic": i} for i in range(5)},
        candidate_positions_sha256={str(i): str(i) * 64 for i in range(5)},
    )
    extracted = extract_survivors(result, 3)
    assert [p["proposal_id"] for p in extracted["survivors"]] == ["2", "3", "4"]
    assert all(len(p["short_steps"]) == 4 for p in extracted["survivors"])
    extracted["survivors"][0]["proposal"]["synthetic"] = 99
    assert result["proposals"]["2"]["synthetic"] == 2
    with pytest.raises(ValueError, match="survivor count"):
        extract_survivors(result, 2)
    bad = copy.deepcopy(result)
    bad["selection"]["events"][2]["steps"][1]["candidate"]["logit_kl"] = 0.3
    with pytest.raises(ValueError):
        extract_survivors(bad, 3)


@pytest.mark.parametrize("outcome", ["accepted", "all_failed", "prefix_drift", "control_failed"])
def test_extensions_require_both_controls_and_keep_32_token_acceptance(monkeypatch, outcome):
    import mosaickv.evaluation.head_substitution_followup as module

    full, anchor, attention = fixture()
    proposals = substitution_pool(full, attention, anchor, prompt_bytes=384, limit=3)
    specs = []
    for i, p in enumerate(proposals):
        steps = null_rows()[:4]
        for r in steps:
            r["candidate"]["logit_kl"] = 0.05
            r["candidate"]["maximum_logit_difference"] = 0.4
        specs.append(
            dict(
                proposal_id=str(i),
                proposal=vars(p),
                positions_sha256=packed_positions_sha(pack_substitution(full, anchor, p)),
                short_steps=steps,
            )
        )
    calls = []

    class Session:
        def __init__(self, *_a, **kwargs):
            assert kwargs["capture_fp64"]

        def sync(self):
            pass

        def probe(self, _packed, *, label):
            calls.append(label)
            rows = null_rows()
            if label.startswith("null_"):
                if outcome == "control_failed" and label == "null_clone":
                    rows[-1]["candidate"]["bitwise_equal_to_anchor_logits"] = False
            else:
                for r in rows:
                    r["candidate"]["logit_kl"] = 0.05
                    r["candidate"]["maximum_logit_difference"] = 0.4
                if label != "0" or outcome in ("all_failed", "prefix_drift"):
                    rows[4]["candidate"]["logit_kl"] = 0.2
                if outcome == "prefix_drift":
                    rows[0]["candidate"]["logit_kl"] += 1e-10
            yield from rows

        def finish(self):
            return dict(measured_source_cache_unchanged=True, logical_positions_unchanged=True)

    monkeypatch.setattr(module, "HeadProbeSession", Session)
    runtime = object.__new__(HeadSubstitutionFollowup)
    runtime.policy = HeadDiagnosticPolicy()
    runtime.config = SimpleNamespace(bounded_decode=SimpleNamespace(sink_tokens=1))
    runtime.adapter = None
    runtime.saved_logits = {}
    runtime.followup = dict(anchor_positions_sha256=packed_positions_sha(anchor), survivors=specs)
    budget = SimpleNamespace(cap_bytes=512, prompt_budget_bytes=384, reserved_generated_bytes=128)
    with pytest.raises(PrefillDiagnosticComplete) as complete:
        runtime._observe_exact_anchor(None, full, anchor, attention, budget)
    record = complete.value.record
    assert calls[:2] == ["null_clone", "null_repack"]
    assert not record["fresh_cohort_authorized"] and not record["calibration_authorized"]
    if outcome == "control_failed":
        assert len(calls) == 2 and record["extensions"] == []
        assert not record["controls_passed"]
    else:
        assert calls[2:] == ["0", "1", "2"]
        assert record["extensions"][0]["accepted"] == (outcome == "accepted")
        expected_lengths = [32, 5, 5] if outcome == "accepted" else [5, 5, 5]
        assert [len(e["steps"]) for e in record["extensions"]] == expected_lengths
        assert all(e["original_four_prefixes_replayed"] for e in record["extensions"]) == (
            outcome != "prefix_drift"
        )
    assert record["existing_pool_exhausted"] == (outcome == "all_failed")
    assert record["draft_conditioned_design_review_indicated"] == (outcome == "all_failed")
    assert not record["selection_modified_by_probe"]
