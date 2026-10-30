import pytest
import torch

from mosaickv.evaluation.dual_projection import output_slots


def test_head_local_source_inverse_and_missing():
    union = torch.tensor([0, 2, 5, 9], dtype=torch.int32)
    selected = torch.tensor([[0, 3, 5], [2, 5, 9]], dtype=torch.int32)
    actual = output_slots(union, selected)
    assert actual.dtype == torch.int16
    assert actual.tolist() == [[0, -1, 2, -1], [-1, 0, 1, 2]]


def test_shared_map_has_one_true_owner():
    union = torch.tensor([0, 2, 5], dtype=torch.int32)
    selected = union[None].expand(32, -1)
    actual = output_slots(union, selected)
    assert actual.stride(0) == 0
    assert actual.untyped_storage().nbytes() == 6
    assert actual[31].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "union,selected", [([1, 0], [[0, 1]]), ([], [[0]]), ([0, 1], [[1, 1]]), ([-1, 0], [[0, 1]])]
)
def test_rejects_ambiguous_maps(union, selected):
    with pytest.raises(ValueError):
        output_slots(torch.tensor(union), torch.tensor(selected))


def test_uncorrected_constructor_measurement_precedes_two_patch_application(monkeypatch):
    from types import SimpleNamespace

    import mosaickv.evaluation.dual_projection as module
    import mosaickv.evaluation.dual_projection_kernel as kernel

    monkeypatch.setattr(module, "decoded", lambda _: torch.zeros(1, 1, 3, 256))
    monkeypatch.setattr(kernel, "project", lambda *args: [t.fill_(2) for t in args[-1]])
    seen = []
    monkeypatch.setattr(kernel, "apply_patch", lambda patch, target: seen.append(patch))
    archive = SimpleNamespace(source=None)
    f = module.DualFormat("dual_projection", (1, 2, 3, 128), archive=archive, output_slots=True)
    targets = tuple(torch.zeros(f.shape) for _ in range(2))
    f.fill(None, targets, direct=True)
    assert all(bool((t == 2).all()) for t in targets)
    assert not seen
    from dataclasses import replace

    replace(f, patches=("key", "value")).fill(None, targets, direct=True)
    assert seen == ["key", "value"]
