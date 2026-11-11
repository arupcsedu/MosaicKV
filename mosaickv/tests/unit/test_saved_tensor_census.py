"""Exact census equivalence and unchanged frozen four-prefix model arithmetic."""

import ast
import gc
import inspect
import subprocess
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import mosaickv.evaluation.draft_logit_source_scores as scoring
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import owner

BASE_SHA = "6ea20f889d3e9abf0b736157827bffbd4b0c3d93"


class LegacySavedTensors:
    """The immutable V22 census, retained only in a CPU test reference."""

    def __init__(self, model):
        self.excluded = {owner(p) for p in model.parameters()}
        self.refs, self.pack_calls, self.peak = {}, 0, 0

    def _observe(self):
        self.refs = {key: ref for key, ref in self.refs.items() if ref() is not None}
        live = tuple(
            t
            for ref in self.refs.values()
            if (t := ref()) is not None and owner(t) not in self.excluded
        )
        self.peak = max(self.peak, backing_bytes(live))

    def pack(self, tensor):
        value = tensor.detach()
        self.refs[id(value)] = weakref.ref(value)
        self.pack_calls += 1
        self._observe()
        return value

    def unpack(self, tensor):
        self._observe()
        return tensor

    def close(self):
        self.refs.clear()


def legacy_current(audit):
    audit._observe()
    return backing_bytes(
        tuple(
            t
            for ref in audit.refs.values()
            if (t := ref()) is not None and owner(t) not in audit.excluded
        )
    )


def test_exact_backing_metric_for_aliases_excluded_weights_and_arbitrary_release_order():
    model = torch.nn.Linear(8, 8, bias=False)
    original, current = LegacySavedTensors(model), scoring._SavedTensors(model)
    backing = torch.empty(256)
    sources = [backing[:3], backing[101:105], torch.empty(17), model.weight, backing[77:80]]
    kept_old, kept_new = [], []
    for source in sources:
        kept_old.append(original.pack(source))
        kept_new.append(current.pack(source))
        assert original.peak == current.peak
        assert legacy_current(original) == current.live_bytes
    assert (
        current.peak == backing.untyped_storage().nbytes() + sources[2].untyped_storage().nbytes()
    )
    assert len(current.refs) == 4  # Excluded weights are never registered.
    assert len(current.owner_counts) == 2
    for i in (1, 0, 4, 3, 2):
        kept_old[i], kept_new[i] = None, None
        assert legacy_current(original) == current.live_bytes
    assert current.live_bytes == 0 and not current.refs and not current.owner_counts
    assert original.peak == current.peak
    current.close()


@pytest.mark.parametrize("count", [100, 500, 1000])
def test_owner_inspection_is_linear_even_with_live_views_and_repeated_weights(count):
    model = torch.nn.Linear(8, 8, bias=False)
    audit = scoring._SavedTensors(model)
    calls = 0

    def counted_owner(tensor):
        nonlocal calls
        calls += 1
        return owner(tensor)

    sources = [torch.empty(8) if i % 2 else model.weight for i in range(count)]
    with patch.object(scoring, "owner", counted_owner):
        packed = [audit.pack(t) for t in sources]
        for value in packed:
            assert audit.unpack(value) is value
    assert calls == count
    assert audit.pack_calls == count
    assert len(audit.refs) == count // 2
    assert audit.peak == (count // 2) * 8 * 4
    audit.close()


def test_weak_callbacks_do_not_retain_tensors_observer_or_reference_cycles():
    model = torch.nn.Linear(8, 8, bias=False)
    audit = scoring._SavedTensors(model)
    value = audit.pack(torch.empty(17))
    value_ref, audit_ref = weakref.ref(value), weakref.ref(audit)
    assert audit.live_bytes == 17 * 4
    del value
    assert value_ref() is None
    assert audit.live_bytes == 0 and not audit.refs
    survivor = audit.pack(torch.empty(29))
    del audit
    # The saved view remains alive but its callback has no strong observer edge.
    assert audit_ref() is None
    del survivor
    gc.collect()
    assert audit_ref() is None


def test_stale_id_callback_cannot_retire_a_replacement_registration():
    audit = scoring._SavedTensors(torch.nn.Linear(8, 8, bias=False))
    view = audit.pack(torch.empty(17))
    reference, key = audit.refs[id(view)]
    other = torch.empty(2)
    unrelated_reference = weakref.ref(other)
    before = audit.live_bytes
    audit._released(id(view), unrelated_reference, key)
    assert audit.refs[id(view)][0] is reference
    assert audit.live_bytes == before
    audit._released(id(view) + 1, reference, key)
    assert audit.live_bytes == before
    audit.close()


def test_close_is_complete_idempotent_and_late_callbacks_are_harmless():
    audit = scoring._SavedTensors(torch.nn.Linear(8, 8, bias=False))
    value = audit.pack(torch.empty(17))
    peak = audit.peak
    audit.close()
    audit.close()
    assert not audit.refs and not audit.owner_counts and audit.live_bytes == 0
    assert audit.peak == peak
    del value
    assert audit.live_bytes == 0
    with pytest.raises(RuntimeError, match="already closed"):
        audit.pack(torch.empty(1))


def test_resized_saved_storage_is_explicitly_unsupported_not_a_false_metric_pass():
    audit = scoring._SavedTensors(torch.nn.Linear(8, 8, bias=False))
    base = torch.empty(17)
    view = audit.pack(base)
    base.resize_(4096)
    with pytest.raises(RuntimeError, match="storage changed"):
        audit.unpack(view)
    audit.close()


def test_dead_parameter_owner_collision_does_not_exclude_new_tensor_storage():
    model = torch.nn.Linear(8, 8, bias=False)
    audit = scoring._SavedTensors(model)
    stale_owner = next(iter(audit.excluded))
    parameter = weakref.ref(model.weight)
    del model
    gc.collect()
    assert parameter() is None
    # Deterministically simulate allocator identity reuse instead of depending
    # on one CPU malloc sequence or the ordering of the full test suite.
    with patch.object(scoring, "owner", lambda _tensor: stale_owner):
        base = torch.empty(17)
        view = audit.pack(base)
    assert id(view) in audit.refs
    assert audit.live_bytes == base.untyped_storage().nbytes()
    base.resize_(4096)
    with pytest.raises(RuntimeError, match="storage changed"):
        audit.unpack(view)
    audit.close()


def test_cpu_cache_gradients_logits_scores_and_peak_are_identical_to_legacy_census():
    # Reuse the fixed random architecture fixture, never load a checkpoint/data.
    from test_draft_logit_source_scores import fixture

    adapter, prefill, packets, pairs = fixture()
    with patch.object(scoring, "_SavedTensors", LegacySavedTensors):
        old_scores, old_report = scoring.draft_row_scores(adapter, prefill, packets, pairs)
    new_scores, new_report = scoring.draft_row_scores(adapter, prefill, packets, pairs)
    assert all(torch.equal(old_scores[k], new_scores[k]) for k in old_scores)
    for key in (
        "reference_logits_sha256",
        "parent_logits_sha256",
        "reference_logits_fp32",
        "parent_logits_fp32",
        "objective_value",
        "positive_row_gains",
        "negative_row_gains",
        "zero_row_gains",
        "nonzero_gradient_scalars",
        "autograd_saved_nonweight_peak_backing_bytes",
        "autograd_saved_tensor_observations",
    ):
        assert old_report[key] == new_report[key]


def test_model_math_and_policy_AST_are_identical_to_frozen_V22_except_observer_close():
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{BASE_SHA}:mosaickv/src/mosaickv/evaluation/draft_logit_source_scores.py",
        ],
        text=True,
    )
    frozen = ast.parse(source)
    current = ast.parse(inspect.getsource(scoring))

    class NormalizeClose(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node) == "audit.refs.clear()":
                return ast.parse("audit.close()", mode="eval").body
            return self.generic_visit(node)

    def functions(tree):
        return {
            node.name: ast.dump(NormalizeClose().visit(node), include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

    assert functions(current) == functions(frozen)
    assert (scoring.HORIZON, scoring.KL_SCALE, scoring.MARGIN_SCALE) == (4, 0.02, 1.0)
