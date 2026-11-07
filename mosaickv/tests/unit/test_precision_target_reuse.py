from unittest.mock import patch

import pytest
import torch

from mosaickv.evaluation import precision_transaction_study as study


def payload(seed):
    g = torch.Generator().manual_seed(seed)
    original = tuple(torch.randn(64, 128, generator=g).bfloat16() for _ in range(2))
    parent = tuple(
        (t.float() + 0.03 * torch.randn(64, 128, generator=g)).bfloat16() for t in original
    )
    return dict(
        original_pair=original,
        parent_pair=parent,
        prompt_queries=torch.randn(16, 128, generator=g).bfloat16(),
        phases=(torch.ones(64, 128).bfloat16(), torch.zeros(64, 128).bfloat16()),
        evidence=dict(
            selected_extra=[2, 3, 4, 5],
            prototype_members=[12, 13],
            original_protected_positions=[0, 1, 63],
            prototype_anchor_position=14,
            prototype_representation="parent_residual_delta",
        ),
        overlay=dict(
            member_keys=torch.zeros(2, 128).bfloat16(),
            prototype_value=torch.zeros(1, 128).bfloat16(),
        ),
    )


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize(
    "mode,skip",
    (("exact", False), ("prototype", False), ("prototype", True), ("joint_guarded", False)),
)
def test_cached_targets_preserve_every_metric_weight_selection_and_veto(mode, skip):
    p = payload(91)
    results, counts = [], []
    for reuse in (False, True):
        with patch.object(study, "attend", wraps=study.attend) as calls:
            results.append(
                study.plan_precision_transaction(
                    p, mode=mode, skip_discarded_exact=skip, reuse_targets=reuse
                )
            )
            counts.append(calls.call_count)
    same(results[0], results[1])
    assert counts[1] < counts[0]


def test_phase_instrumentation_cannot_change_prototype_membership():
    from test_storage_error_aware import arguments

    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy, plan_layer

    args, parent = arguments()
    overlays = [
        plan_layer(
            *args,
            OverlayPolicy(record_phase_timings=timing, prototype_anchor_scope="selected"),
            pin_residuals=False,
            parent_pair=parent,
        )
        for timing in (False, True)
    ]
    assert overlays[0].evidence["selected_extra"] == overlays[1].evidence["selected_extra"]
    assert overlays[0].evidence["prototype_members"] == overlays[1].evidence["prototype_members"]
    # Scratch buffers are intentionally uninitialized; compare actual reads,
    # not arbitrary bytes that are overwritten before their first use.
    reads = []
    for overlay in overlays:
        pair = tuple(t.clone() for t in parent)
        for component, tensor in enumerate(pair):
            overlay.fill(component, tensor)
        reads.append(pair)
    for a, b in zip(*reads, strict=True):
        assert torch.equal(a, b)
