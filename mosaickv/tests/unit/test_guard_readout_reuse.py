import pytest
import torch

from mosaickv.evaluation.guard_readout_reuse import ReadoutReuse
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


@pytest.mark.parametrize("passed", [True, False])
def test_cached_selected_native_readout_is_exact_and_consumed_once(passed):
    norm, head = torch.nn.LayerNorm(4096), torch.nn.Linear(4096, 19, bias=False)
    parent, active = torch.randn(1, 1, 4096), torch.randn(1, 1, 4096)
    original_norm, original_head = norm.forward, head.forward
    g = ReadoutReuse(norm, head)
    with torch.no_grad(), g:
        pn, an = norm(parent), norm(active)
        pl, al = head(pn), head(an)
        g.stage(torch.tensor(passed), pn, an, pl, al)
        hidden = active if passed else parent
        assert torch.equal(head(norm(hidden)), al if passed else pl)
        assert g.pending_norm is g.pending_logits is None
        assert g.report()["normalization_reuses"] == g.report()["vocabulary_reuses"] == 1
        assert g.report()["peak_temporary_activation_bytes"] == (4096 + 19) * 4
    assert norm.forward == original_norm and head.forward == original_head


def test_ordering_nonconsumption_and_failure_cannot_leak_cached_outputs():
    norm, head = torch.nn.Identity(), torch.nn.Linear(4096, 19, bias=False)
    x, z = torch.zeros(1, 1, 4096), torch.zeros(1, 1, 19)
    g = ReadoutReuse(norm, head)
    with pytest.raises(RuntimeError, match="exactly once"), g:
        g.stage(torch.tensor(True), x, x, z, z)
        with pytest.raises(RuntimeError, match="precede"):
            head(x)
        with pytest.raises(RuntimeError, match="not consumed"):
            g.stage(torch.tensor(True), x, x, z, z)
    assert g.pending_norm is g.pending_logits is None
    with pytest.raises(ValueError), ReadoutReuse(norm, head) as g:
        g.stage(torch.tensor(True), x, x, z, z)
        norm(x[:, :, :4])
    assert norm(x) is x
    with pytest.raises(ValueError, match="readout reuse"):
        OverlayPolicy(reuse_anchor_readout=True)
