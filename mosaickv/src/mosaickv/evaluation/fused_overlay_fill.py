"""Default-off fused reader with identical existing precision decisions."""

import torch


def fill(overlay, component, destination):
    if (
        component not in (0, 1)
        or overlay.component_mode != "full"
        or overlay.prototype_representation not in ("absolute_kv", "parent_residual_delta")
        or destination.device.type != "cuda"
        or destination.dtype != torch.bfloat16
        or destination.ndim != 4
        or destination.shape[0] != 1
        or destination.shape[-1] != 128
        or destination.stride(-1) != 1
        or not 0 <= overlay.head < destination.shape[1]
        or overlay.mixed
    ):
        raise ValueError("isolated full BF16 CUDA overlay with no mixed scratch required")
    sources = (*overlay.gpu_tensors(),)
    if any(t.device != destination.device or not t.is_contiguous() for t in sources):
        raise ValueError("same-device contiguous immutable overlay payloads required")
    import triton

    from mosaickv.evaluation.fused_overlay_kernel import update

    exact, members = overlay.exact_ids.numel(), overlay.positions.numel()
    update[(1,)](
        destination,
        overlay.exact_ids,
        overlay.exact_pair[component],
        overlay.positions,
        overlay.member_keys,
        overlay.prototype_pair[1],
        overlay.promotion[component],
        overlay.promoted,
        overlay.actual_fills,
        HEAD_OFFSET=overlay.head * destination.stride(1),
        ROW_STRIDE=destination.stride(-2),
        EXACT=exact,
        MEMBERS=members,
        DIM=128,
        KEY=component == 0,
        DELTA=overlay.prototype_representation == "parent_residual_delta",
        BLOCK=triton.next_power_of_2((exact + members) * 128),
    )


def native_controls():
    """Synthetic correctness only; never a measured model-quality row."""
    from mosaickv.evaluation.storage_six_overlay import LayerOverlay

    rows = []
    generator = torch.Generator().manual_seed(31)

    def values(n):
        return torch.randn(n, 128, generator=generator).bfloat16().cuda()

    for representation in ("absolute_kv", "parent_residual_delta"):
        for promoted in (False, True):
            args = dict(
                head=1,
                positions=(2, 4),
                exact_ids=(8, 10),
                exact_pair=(values(2), values(2)),
                prototype_pair=(values(1), values(1)),
                phases=(torch.ones(3, 128, device="cuda"), torch.zeros(3, 128, device="cuda")),
                residual_pair=(values(2), values(2)),
                anchor=2,
                dispersion=0.2,
                evidence={},
            )
            a, b = LayerOverlay(**args), LayerOverlay(**args, fill_backend="fused")
            a.prototype_representation = b.prototype_representation = representation
            if promoted:
                a.promote()
                b.promote()
            for component in (0, 1):
                x = values(2 * 17).reshape(1, 2, 17, 128)
                y = x.clone()
                a.fill(component, x[..., :13, :])
                b.fill(component, y[..., :13, :])
                torch.cuda.synchronize()
                if not torch.equal(x.view(torch.int16), y.view(torch.int16)):
                    raise ValueError(
                        "fused fill changed K/V words, other heads or trailing history"
                    )
                if not torch.equal(a.actual_fills, b.actual_fills):
                    raise ValueError("fused fill changed committed fill accounting")
                rows.append(
                    dict(
                        representation=representation,
                        promoted=promoted,
                        component=component,
                        complete_strided_backing_bitwise=True,
                    )
                )
    return dict(passed=True, measurement_type="synthetic_native_correctness_not_quality", rows=rows)
