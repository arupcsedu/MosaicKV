from __future__ import annotations

import json

import pytest

from mosaickv.selection import (
    CalibrationStabilityError,
    FidelityObservation,
    HeadFidelityObservation,
    derive_adaptive_head_routing,
    select_fidelity_profile,
)


def _rows() -> tuple[FidelityObservation, ...]:
    return tuple(
        FidelityObservation(variant, sample, 100 if variant == "safe" else 80, *metrics)
        for variant, metrics in (
            ("safe", (1.0, 0.01, 0.02, 0.01)),
            ("lossy", (0.5, 0.5, 0.5, 0.5)),
        )
        for sample in ("cal-1", "cal-2")
    )


def test_label_free_fidelity_profile_is_deterministic_and_disjoint(tmp_path: object) -> None:
    profile = select_fidelity_profile(
        _rows(),
        parameters_by_variant={"safe": {"lambda_q": 1.0}, "lossy": {"lambda_q": 0.5}},
        calibration_dataset_id="dataset",
        calibration_dataset_revision="revision",
        calibration_split="train[:2]",
        evaluation_sample_ids=("eval-2", "eval-1"),
        target_active_kv_bytes=100,
    )
    assert profile.selected_variant == "safe"
    assert profile.evaluation_sample_ids == ("eval-1", "eval-2")
    assert len(profile.sha256) == 64
    output = tmp_path / "profile.json"  # type: ignore[operator]
    profile.write(output)
    assert json.loads(output.read_text())["profile_sha256"] == profile.sha256


def test_fidelity_calibration_rejects_evaluation_leakage() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        select_fidelity_profile(
            _rows(),
            parameters_by_variant={"safe": {}, "lossy": {}},
            calibration_dataset_id="dataset",
            calibration_dataset_revision="revision",
            calibration_split="train[:2]",
            evaluation_sample_ids=("cal-1",),
            target_active_kv_bytes=100,
        )


def test_bootstrap_stability_gate_rejects_unstable_profile() -> None:
    rows = tuple(
        FidelityObservation(variant, sample, 100, agreement, 0.0, 0.0, 0.0)
        for variant, agreements in (
            ("left", (1.0, 0.0)),
            ("right", (0.0, 1.0)),
        )
        for sample, agreement in zip(("cal-1", "cal-2"), agreements, strict=True)
    )
    with pytest.raises(CalibrationStabilityError, match="bootstrap stability") as raised:
        select_fidelity_profile(
            rows,
            parameters_by_variant={"left": {}, "right": {}},
            calibration_dataset_id="dataset",
            calibration_dataset_revision="revision",
            calibration_split="validation",
            evaluation_sample_ids=("eval",),
            target_active_kv_bytes=100,
            bootstrap_resamples=100,
            bootstrap_seed=7,
            minimum_selection_frequency=0.8,
        )
    assert raised.value.required_frequency == 0.8
    assert 0 <= raised.value.best_frequency < 0.8


def test_bootstrap_does_not_reapply_full_ledger_byte_ceiling() -> None:
    rows = tuple(
        FidelityObservation(
            "only",
            sample,
            active_bytes,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        for sample, active_bytes in (("cal-1", 90), ("cal-2", 110))
    )
    profile = select_fidelity_profile(
        rows,
        parameters_by_variant={"only": {}},
        calibration_dataset_id="dataset",
        calibration_dataset_revision="revision",
        calibration_split="validation",
        evaluation_sample_ids=("eval",),
        target_active_kv_bytes=100,
        bootstrap_resamples=100,
        bootstrap_seed=7,
        minimum_selection_frequency=1.0,
    )
    assert profile.bootstrap_selection_frequency == 1.0


def test_adaptive_head_routing_preserves_worst_heads_exact() -> None:
    rows = tuple(
        HeadFidelityObservation(
            sample_id=sample,
            layer=0,
            kv_head=head,
            exact_error=0.1,
            prototype_error=prototype_error,
        )
        for sample in ("cal-1", "cal-2")
        for head, prototype_error in ((0, 0.5), (1, 0.2), (2, 0.1), (3, 0.1))
    )
    routing = derive_adaptive_head_routing(
        rows,
        default_group_size=15,
        sensitive_group_size=11,
        sensitive_position_span=127,
        damage_threshold=0.05,
        exact_head_fraction=0.25,
    )
    assert routing.exact_head_routes == ("0:0",)
    assert routing.head_group_sizes == ("0:1:11",)
    assert routing.head_position_spans == ("0:1:127",)
