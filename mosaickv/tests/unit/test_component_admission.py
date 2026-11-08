import pytest

from mosaickv.evaluation.component_admission import SIX_COMPONENTS, component_scope
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def test_source_only_variant_never_installs_inactive_overlay_hooks():
    from mosaickv.evaluation import storage_runs, streaming_source_constructor
    from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer

    original = (
        storage_runs.run,
        storage_runs.cache_from_formats,
        streaming_source_constructor.StreamingDirectSource,
        SequentialSourceHistoryLayer.materialize_component,
    )
    policy = OverlayPolicy(enabled=False, quality_objective="bitwise_storage_only")
    with component_scope(object(), policy, mode="storage_preservation") as report:
        assert original == (
            storage_runs.run,
            storage_runs.cache_from_formats,
            streaming_source_constructor.StreamingDirectSource,
            SequentialSourceHistoryLayer.materialize_component,
        )
        assert report["disabled_components"] == list(SIX_COMPONENTS)
        assert report["original_six_mechanism_design_active"] is False
        assert report["extra_gpu_bytes"] == report["residual_cpu_bytes"] == 0
        assert not report["quality_improvement_claim_authorized"]


def test_experimental_overlay_remains_explicit_and_restores_on_error():
    from mosaickv.evaluation import storage_runs

    original = storage_runs.run
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        component_scope(None, OverlayPolicy(enabled=False), mode="experimental_overlay"),
    ):
        assert storage_runs.run is not original
        raise RuntimeError("deliberate")
    assert storage_runs.run is original


def test_no_silent_switch_of_enabled_or_unverified_source_policies():
    for policy, mode in (
        (OverlayPolicy(), "storage_preservation"),
        (OverlayPolicy(enabled=False), "storage_preservation"),
        (
            OverlayPolicy(
                enabled=False, quality_objective="bitwise_storage_only", replace_source_codes=True
            ),
            "storage_preservation",
        ),
        (OverlayPolicy(enabled=False), "auto"),
    ):
        with pytest.raises(ValueError), component_scope(None, policy, mode=mode):
            pass
