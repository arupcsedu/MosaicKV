import pytest
import torch

from mosaickv.evaluation.eager_core_operator import Core, controls
from mosaickv.evaluation.uniform_eager_core import strategy, supported


def test_uniform_capability_is_selected_by_geometry_not_method_or_labels():
    assert strategy((600,) * 32, 127) == "uniform_split_graph"
    assert supported((300,) * 32, 127)  # Uniform baseline gets the SAME optimization.


def test_adaptive_layout_is_unchanged_original_not_the_failed_out_fallback():
    assert strategy(tuple(range(300, 332)), 127) == "original_eager_nonuniform"
    assert not supported(tuple(range(300, 332)), 127)


@pytest.mark.parametrize("lengths,updates", [((), 127), ((0,) * 32, 127), ((600,) * 32, 128)])
def test_invalid_metadata_fails_closed(lengths, updates):
    with pytest.raises(ValueError, match="registered"):
        strategy(lengths, updates)


def test_unverified_geometry_is_an_explicit_no_acceleration():
    assert strategy((600,) * 16, 127) == "original_eager_ineligible_geometry"
    assert strategy((1000,) * 32, 127) == "original_eager_ineligible_geometry"


def test_cuda_control_geometry_is_checked_before_device_work():
    with pytest.raises(ValueError, match="physical lease"):
        controls(lengths=(600,), capacity=431)


def test_view_descriptor_reuse_aliases_only_the_same_fixed_owners():
    lease = torch.zeros(1, 2, 17, 128, dtype=torch.bfloat16)
    core = Core(lease, mask_dtype=None, graph=False, frequencies={}, reuse_views=True)
    first = core.views(7)
    assert core.views(7) is first
    assert core.views(13) is not first
    assert core.view_builds == 2 and core.view_reuses == 1
    assert first.scores.untyped_storage().data_ptr() == core.scores.untyped_storage().data_ptr()
    assert first.lease.untyped_storage().data_ptr() == lease.untyped_storage().data_ptr()
    core.close()
    assert not core.view_cache
