import subprocess
import sys

import evicastkv_fused_predicate_screen as control
import pytest
import torch

from mosaickv.evaluation import deferred_transport_checks as validation
from mosaickv.evaluation import fused_transport_predicate as fused
from mosaickv.evaluation.shared_source_format import transport_corrections


def mock_fused(reference, actual, *, out):
    bad, finite = fused.original(reference, actual)
    out.copy_(bad.flatten())
    return out, finite


def test_default_scope_inactive_and_exception_cannot_leak():
    assert not fused.active()
    with pytest.raises(LookupError), fused.use(True) as state:
        assert fused.active()
        raise LookupError()
    assert state.closed and not fused.active()
    with fused.use(False) as state:
        assert state is None and not fused.active()
        with pytest.raises(ValueError, match="nonnested"), fused.use(True):
            pass


def test_mocked_cpu_transaction_retains_canonical_patch_words(monkeypatch):
    # Tests orchestration only. Real CUDA equivalence is a separate operator gate.
    monkeypatch.setattr(fused, "fused", mock_fused)
    expected = torch.arange(256, dtype=torch.float32).bfloat16().reshape(1, 1, 2, 128)
    actual = expected.clone()
    actual.flatten()[3] = 8
    protected = torch.tensor([1], dtype=torch.int32)
    old = transport_corrections((expected,), (actual,), protected)[0]
    with (
        validation.use(True, coalesce_nonzero=True, full_predicate_span=True),
        fused.use(True) as state,
    ):
        new = transport_corrections((expected,), (actual,), protected)[0]
    assert torch.equal(new.indices, old.indices)
    assert torch.equal(new.values.view(torch.int16), old.values.view(torch.int16))
    assert state.report()["calls"] == 1
    assert state.report()["closed_before_decode"]
    assert state.report()["retained_tensor_owners"] == 0


@pytest.mark.parametrize("failure", ["finite", "protected"])
def test_original_failure_predicates_still_abort_at_commit(monkeypatch, failure):
    monkeypatch.setattr(fused, "fused", mock_fused)
    expected = torch.ones(1, 1, 2, 128, dtype=torch.bfloat16)
    actual = expected.clone()
    actual.flatten()[0] = float("inf") if failure == "finite" else 2
    with (
        pytest.raises((ValueError, RuntimeError), match=failure),
        validation.use(True, coalesce_nonzero=True, full_predicate_span=True),
        fused.use(True),
    ):
        transport_corrections((expected,), (actual,), torch.tensor([0]))
    assert not fused.active()


def test_incompatible_mask_does_not_silently_fall_back():
    expected = torch.zeros(8, dtype=torch.bfloat16)
    with fused.use(True):
        with pytest.raises(ValueError, match="full-span"):
            fused.fill_mask(expected, expected, None, quantum=8)
        with pytest.raises(ValueError, match="full-span"):
            fused.fill_mask(expected, expected, torch.empty(8, dtype=torch.bool), quantum=4)


def test_variant_changes_only_predicate_and_offers_noop_to_baselines():
    previous = control.OLD_OVERRIDES("mosaickv_exact", "joint-async-handoff")
    new = control.overrides("mosaickv_exact", "joint-async-fused-predicate")
    assert {k: v for k, v in new.items() if k != "fused_transport_predicate"} == previous
    assert new["fused_transport_predicate"] is True
    assert control.overrides("prefixkv_reimpl", "graph-history-graph")["fused_transport_predicate"]


def test_post_operator_kernel_or_bound_changes_cannot_use_old_proof():
    source = "LIMIT = 32\ndef _mask(x):\n return x + 1\ndef validate(x):\n return x > 0\n"
    same = source + "\ndef accounting_only():\n return 0\n"
    assert control.arithmetic_contract(source) == control.arithmetic_contract(same)
    assert control.arithmetic_contract(source) != control.arithmetic_contract(
        source.replace("x + 1", "x + 2")
    )
    assert control.arithmetic_contract(source) != control.arithmetic_contract(
        source.replace("LIMIT = 32", "LIMIT = 64")
    )
    with pytest.raises(ValueError, match="complete bounded"):
        control.arithmetic_contract("LIMIT = 32")


def test_real_launcher_configuration_routes_all_four_arms_without_jobs():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.argv = ['controller']
import evicastkv_fused_predicate_screen as c
def dispatch():
    records = {
        'mosaickv_exact': {'graph-lt': {'packet_sha256': 'source'}},
        'prefixkv_reimpl': {a: {'packet_sha256': a} for a in ('graph', 'graph-original')},
    }
    values = c.base.control.configurations(records)
    assert tuple(values) == c.ARMS
    assert c.base.control.CANDIDATE_ARM == 'joint-async-fused-predicate'
    assert c.base.control.prepare is c.prepare
    assert c.base.control.summary is c.summary
    for (method, arm), value in values.items():
        assert value['fused_transport_predicate'] is (
            arm == 'joint-async-fused-predicate' or method != 'mosaickv_exact'
        )
c.prior.prior.OLD_MAIN = dispatch
c.main()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
