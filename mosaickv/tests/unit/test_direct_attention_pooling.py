import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mosaickv.graph.pooling import _pool_attention_block_batch


@pytest.mark.parametrize("heads,width", [(1, 1), (2, 4), (3, 9)])
def test_direct_coordinates_match_original_repeated_source_and_reduction_bitwise(heads, width):
    generator = torch.Generator().manual_seed(heads * 100 + width)
    grouped = torch.rand(heads, 16, 37, generator=generator)
    positions = tuple(tuple(range(i, i + width)) for i in range(8))
    head_ids = tuple(i % heads for i in range(8))
    source = grouped[torch.tensor(head_ids)]
    expected = torch.gather(source, 2, torch.tensor(positions)[:, None, :].expand(-1, 16, -1)).mean(
        2
    )
    assert torch.equal(
        expected, _pool_attention_block_batch(grouped, positions=positions, heads=head_ids)
    )


def test_one_token_blocks_never_materialize_blocks_times_queries_times_all_keys():
    class Allocations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.sizes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor):
                self.sizes.append(result.numel())
            return result

    n, q = 603, 16
    grouped = torch.ones(1, q, n)
    with Allocations() as allocations:
        pooled = _pool_attention_block_batch(
            grouped, positions=tuple((i,) for i in range(n)), heads=(0,) * n
        )
    assert pooled.shape == (n, q)
    assert max(allocations.sizes) <= n * q


def test_negative_indices_are_rejected_instead_of_wrapping():
    grouped = torch.ones(1, 3, 7)
    for positions, heads in ((((-1,),), (0,)), (((7,),), (0,)), (((1,),), (-1,))):
        with pytest.raises(ValueError):
            _pool_attention_block_batch(grouped, positions=positions, heads=heads)
