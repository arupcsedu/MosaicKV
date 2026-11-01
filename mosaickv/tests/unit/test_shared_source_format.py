from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.protected_projection_source import protected_positions
from mosaickv.evaluation.shared_source_format import SourceCapture, construct


def example():
    torch.manual_seed(5592)
    h, n, d = 2, 33, 128
    x = torch.randn(1, n, h * d).bfloat16()
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    angles = torch.randn(1, n, 64).repeat(1, 1, 2)
    phases = angles.cos().bfloat16(), angles.sin().bfloat16()
    with torch.inference_mode():
        key = attention.k_proj(x).reshape(1, n, h, d).transpose(1, 2)
        value = attention.v_proj(x).reshape(1, n, h, d).transpose(1, 2)
        key = (
            key * phases[0][:, None]
            + torch.cat((-key[..., 64:], key[..., :64]), -1) * phases[1][:, None]
        )
    memberships = tuple(tuple(sorted({*range(head, 32, 2), 32})) for head in range(2))
    identity = (
        torch.arange(n, dtype=torch.int32),
        torch.arange(n, dtype=torch.int16)[None].expand(h, n),
    )
    text = (0, 1, 31, 32)
    return x, (key, value), memberships, text, phases, identity, attention


@pytest.mark.parametrize("exclude", [False, True])
def test_full_source_positions_exact_rows_storage_isolation_and_history(
    exclude,
):
    x, pair, heads, text, phases, identity, attention = example()
    before = tuple(t.clone() for t in (*pair, x))
    fmt = construct(x, pair, heads, text, phases, identity, exclude_global_exact=exclude)
    assert fmt.shape == tuple(pair[0].shape)
    mask = protected_positions(heads, text, length=33)
    targets = tuple(torch.full((1, 2, 49, 128), -71).bfloat16() for _ in range(2))
    fmt.fill(attention, targets, direct=True)
    for out, original in zip(targets, pair, strict=True):
        assert torch.equal(out[0, :, :33][mask], original[0][mask])
        assert (out[..., 33:, :] == -71).all()
        assert torch.isfinite(out).all()
    for old, now in zip(before, (*pair, x), strict=True):
        assert torch.equal(old, now)
    assert fmt.component_bytes == backing_bytes(fmt.tensors())


def test_conditional_source_has_identical_storage_and_no_extra_persistent_mask():
    x, pair, heads, text, phases, identity, _ = example()
    a = construct(x, pair, heads, text, phases, identity)
    b = construct(x, pair, heads, text, phases, identity, exclude_global_exact=True)
    assert a.component_bytes == b.component_bytes
    assert tuple(t.shape for t in a.private()) == tuple(t.shape for t in b.private())
    with pytest.raises(ValueError):
        construct(x[..., :128], pair, heads, text, phases, identity)
    with pytest.raises(ValueError):
        construct(x, pair, heads, text, phases, (identity[0].flip(0), identity[1]))


def test_wrong_lease_type_and_incomplete_pair_rejected_before_projection():
    x, pair, heads, text, phases, identity, attention = example()
    fmt = construct(x, pair, heads, text, phases, identity)
    with pytest.raises(ValueError, match="complete layer lease"):
        fmt.fill(attention, (pair[0],), direct=True)
    with pytest.raises(ValueError, match="complete layer lease"):
        fmt.fill(attention, tuple(t.float() for t in pair), direct=True)


def test_source_capture_only_prefill_and_hook_cleanup():
    class Attention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    module = Attention()
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=module)])
    x = torch.randn(1, 33, 256).bfloat16()
    with SourceCapture(adapter) as capture:
        module(hidden_states=x)
    assert torch.equal(x, capture.records[0]) and x.data_ptr() != capture.records[0].data_ptr()
    assert not module._forward_pre_hooks


def test_kernel_lease_geometry_is_not_specialized_per_decode_token():
    from mosaickv.evaluation.shared_source_kernel import _project

    parameters = {p.name: p for p in _project.params}
    assert all(
        not parameters[n].is_constexpr and parameters[n].do_not_specialize
        for n in ("ROWS", "HEAD_OUT")
    )
