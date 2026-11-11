"""CPU lifecycle controls only; random tiny models, no pretrained evidence."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from mosaickv.adapters.huggingface.types import PreparedInputs
from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace
from mosaickv.evaluation.bounded_ragged_raw import build_raw_packets
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.streaming_component_prefill import (
    RetiredPrefillLayer,
    StreamingComponentPrefill,
    run,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    torch.manual_seed(27)
    config = LlamaConfig(
        vocab_size=19,
        hidden_size=256,
        intermediate_size=320,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    prepared = PreparedInputs({"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]])}, (), 7)
    adapter = SimpleNamespace(
        model=model,
        _language_layers=lambda: model.model.layers,
        _prefill_forward=lambda args: model(**args),
        _capture_model_state=lambda output: {},
        _greedy_token=lambda logits: logits.argmax(-1, keepdim=True),
    )
    workspace = ComponentWorkspace(device="cpu", heads=2)
    positions = torch.arange(512, dtype=torch.long)
    return adapter, prepared, workspace, positions


def native(adapter, prepared):
    windows = {}
    handles = []
    for index, layer in enumerate(adapter._language_layers()):

        def save(module, args, output, *, index=index):
            windows[index] = output[1][..., -16:, :].detach().clone()

        handles.append(layer.self_attn.register_forward_hook(save))
    try:
        with torch.inference_mode():
            output = adapter.model(
                **prepared.model_inputs,
                past_key_values=DynamicCache(),
                cache_position=torch.arange(prepared.logical_sequence_length),
                use_cache=True,
                return_dict=True,
                output_attentions=False,
                logits_to_keep=1,
            )
        pairs = output.past_key_values.to_legacy_cache()
        return output.logits.clone(), pairs, windows
    finally:
        for handle in handles:
            handle.remove()


def test_metadata_only_retired_layer_refuses_decode():
    layer = RetiredPrefillLayer(7)
    assert layer.keys is layer.values is None
    assert layer.get_seq_length() == 7
    assert layer.get_mask_sizes(torch.arange(3)) == (10, 0)
    assert layer.get_max_cache_shape() == -1
    with pytest.raises(RuntimeError, match="retired"):
        layer.update(None, None)


def test_streaming_source_preserves_real_native_prefill_and_retires_all_views():
    adapter, prepared, workspace, positions = fixture()
    expected, original, _windows = native(adapter, prepared)
    observations = []

    def observe(tensors, phase):
        observations.append((phase, backing_bytes(tensors)))

    prefill, packets, report = run(
        adapter,
        prepared,
        mode="source",
        workspace=workspace,
        positions=positions,
        observer=observe,
    )
    assert torch.equal(expected[:, -1], prefill.logits)
    assert report["maximum_live_native_cache_layers"] == 1
    assert report["native_layers_retired"] == 3
    assert report["all_native_owner_weakrefs_released"] is True
    assert report["uses_original_fullkv_snapshot"] is False
    assert report["outstanding_native_owner_references"] == 0
    assert report["complete_request_peak_certified"] is False
    assert report["constructor_workspace_shared_with_decode"] is True
    assert report["benchmark_labels_read"] is False
    assert report["fallback_used"] is False
    assert report["constructor_temporaries_audited"] is True
    assert report["constructor_audit_layers"] == [0, 1, 2]
    assert all(row["constructor_audit_operations"] > 0 for row in report["packets"])
    assert all(
        row["constructor_audit_observations"] >= row["constructor_audit_operations"]
        for row in report["packets"]
    )
    assert all(
        "torch.bool" in row["observed_same_device_dtypes"] for row in report["constructor_audits"]
    )
    assert all(layer.keys is layer.values is None for layer in prefill.state.past_key_values.layers)
    assert prefill.state.logical_sequence_length == prefill.state.next_decode_position == 7
    assert len(packets) == 3
    assert not any(layer.self_attn._forward_hooks for layer in adapter._language_layers())
    assert not any(layer.self_attn._forward_pre_hooks for layer in adapter._language_layers())
    for index, packet in enumerate(packets):
        target, scratch = workspace.component_view(7), workspace.head
        for component in (0, 1):
            packet.fill_component(
                adapter._language_layers()[index].self_attn, component, target, rope_scratch=scratch
            )
            assert torch.equal(
                target.view(torch.int16), original[index][component].view(torch.int16)
            )
        phases = {phase for phase, _ in observations}
        assert f"prefill_layer_{index}_source_captured" in phases
        assert f"prefill_layer_{index}_KV_published" in phases
        assert f"prefill_layer_{index}_packet_constructed" in phases
        assert f"prefill_layer_{index}_native_retired" in phases
    assert observations[-1][0] == "prefill_complete"
    assert observations[-1][1] == backing_bytes(
        (*workspace.owned_tensors(), positions, *(t for p in packets for t in p.tensors()))
    )


def test_streaming_raw_matches_frozen_all_layer_positions_values_and_prefill_logits():
    adapter, prepared, workspace, positions = fixture()
    expected, original, windows = native(adapter, prepared)
    target = 21 * 512
    reference, old_report = build_raw_packets(adapter, original, windows, target, (0, 6))
    prefill, packets, report = run(
        adapter,
        prepared,
        mode="raw",
        workspace=workspace,
        positions=positions,
        target_bytes=target,
        mandatory=(0, 6),
    )
    assert torch.equal(expected[:, -1], prefill.logits)
    assert report["native_layers_retired"] == 3
    assert report["raw_score_gpu_window_retained"] is False
    assert report["constructor_temporaries_audited"] is True
    assert report["constructor_audit_layers"] == [0, 1, 2]
    assert report["prompt_packet_bytes"] == old_report["actual_raw_KV_bytes"] == target
    assert report["raw_position_map_sha256"] == old_report["position_map_sha256"]
    for new, old in zip(packets, reference, strict=True):
        assert new.prompt_lengths == old.prompt_lengths
        assert all(
            torch.equal(a, b)
            for a, b in zip(new.original_positions, old.original_positions, strict=True)
        )
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(new.tensors(), old.tensors(), strict=True)
        )


def test_constructor_audit_excludes_original_probability_alias_but_charges_real_clone():
    from mosaickv.evaluation.projection_cache import identity

    adapter, prepared, workspace, positions = fixture()
    ordinary_ids, clone_ids, forbidden_seen = set(), set(), []
    handles = []
    for layer in adapter._language_layers():

        def ordinary(module, args, output):
            # Only the current layer's still-live activation is excluded.
            # Dead earlier activations may share an address with a real clone.
            ordinary_ids.clear()
            ordinary_ids.add(identity(output[1]))

        handles.append(layer.self_attn.register_forward_hook(ordinary))

    def observe(tensors, phase):
        if phase.endswith(("constructor_workspace", "selection_window")):
            current = {identity(t) for t in tensors}
            forbidden_seen.extend(current & ordinary_ids)
            clone_ids.update(identity(t) for t in tensors if t.shape == (1, 2, 7, 7))

    try:
        run(
            adapter,
            prepared,
            mode="raw",
            workspace=workspace,
            positions=positions,
            target_bytes=21 * 512,
            mandatory=(0, 6),
            observer=observe,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert ordinary_ids
    assert clone_ids
    assert not forbidden_seen
    # Only contemporaneous alias exclusions matter; a CPU allocator can reuse
    # a dead earlier layer's address for a genuinely new clone in a later layer.


def test_raw_method_specific_selection_buffers_are_charged_before_they_are_released():
    adapter, prepared, workspace, positions = fixture()
    observed = {}

    def observe(tensors, phase):
        observed[phase] = [(tuple(t.shape), t.dtype) for t in tensors]

    run(
        adapter,
        prepared,
        mode="raw",
        workspace=workspace,
        positions=positions,
        target_bytes=21 * 512,
        mandatory=(0, 6),
        observer=observe,
    )
    for index in range(3):
        prefix = f"prefill_layer_{index}_selection_"
        window = ((1, 2, 7, 7), torch.bfloat16)
        float_window = ((1, 2, 7, 7), torch.float32)
        mean = ((2, 7), torch.float32)
        assert window in observed[prefix + "window"]
        assert window in observed[prefix + "float"]
        assert float_window in observed[prefix + "float"]
        assert window not in observed[prefix + "window_retired"]
        assert float_window in observed[prefix + "mean"]
        assert mean in observed[prefix + "mean"]
        assert all(
            item not in observed[prefix + "GPU_buffers_retired"]
            for item in (window, float_window, mean)
        )


def test_observer_retaining_original_native_owners_fails_closed():
    adapter, prepared, workspace, positions = fixture()
    retained = []

    def retain(tensors, phase):
        if phase.endswith("KV_published"):
            retained.extend(tensors)

    with pytest.raises(RuntimeError, match="survived retirement"):
        run(
            adapter,
            prepared,
            mode="source",
            workspace=workspace,
            positions=positions,
            observer=retain,
        )
    assert not any(layer.self_attn._forward_hooks for layer in adapter._language_layers())
    assert not any(layer.self_attn._forward_pre_hooks for layer in adapter._language_layers())


def test_callback_failure_removes_all_hooks_and_restores_private_update():
    adapter, prepared, workspace, positions = fixture()

    def reject(tensors, phase):
        if phase.endswith("source_captured"):
            raise RuntimeError("injected owner observer failure")

    stream = StreamingComponentPrefill(
        adapter, prepared, mode="source", workspace=workspace, positions=positions, observer=reject
    )
    original = stream.cache.update
    with pytest.raises(RuntimeError, match="injected"), stream:
        stream.prefill()
    assert stream.cache.update == original
    assert not any(layer.self_attn._forward_hooks for layer in adapter._language_layers())
    assert not any(layer.self_attn._forward_pre_hooks for layer in adapter._language_layers())


def test_retirement_receipts_follow_real_tensor_death_and_precede_next_capture():
    adapter, prepared, workspace, positions = fixture()
    events = []

    class Tracker:
        def __init__(self):
            self.references = {}

        def capture(self, pair, *, label):
            events.append(("capture", label))
            self.references[label] = tuple(weakref.ref(tensor) for tensor in pair)
            assert all(reference() is not None for reference in self.references[label])
            return dict(label=label)

        def verify(self, token, *, synchronize):
            assert synchronize is True
            label = token["label"]
            assert all(reference() is None for reference in self.references[label])
            events.append(("verify", label))
            return dict(label=label, CPU_fixture_not_physical_evidence=True)

    def observe(tensors, phase):
        if phase.endswith("native_retired"):
            events.append(("retirement_observed", phase))

    _, _, report = run(
        adapter,
        prepared,
        mode="source",
        workspace=workspace,
        positions=positions,
        observer=observe,
        retirement_tracker=Tracker(),
    )
    assert [item for item in events if item[0] in {"capture", "verify"}] == [
        (action, f"layer_{index}") for index in range(3) for action in ("capture", "verify")
    ]
    for index in range(3):
        assert events.index(("verify", f"layer_{index}")) < events.index(
            ("retirement_observed", f"prefill_layer_{index}_native_retired")
        )
    assert report["native_CUDA_free_completed_verified"] is False
    assert len(report["native_retirement_receipts"]) == 3


def test_missing_physical_retirement_receipt_fails_closed_and_cleans_hooks():
    adapter, prepared, workspace, positions = fixture()

    class Tracker:
        def capture(self, pair, *, label):
            return dict(label=label)

        def verify(self, token, *, synchronize):
            raise RuntimeError("missing native FREE_COMPLETED generation")

    with pytest.raises(RuntimeError, match="FREE_COMPLETED"):
        run(
            adapter,
            prepared,
            mode="source",
            workspace=workspace,
            positions=positions,
            retirement_tracker=Tracker(),
        )
    assert not any(layer.self_attn._forward_hooks for layer in adapter._language_layers())
    assert not any(layer.self_attn._forward_pre_hooks for layer in adapter._language_layers())


@pytest.mark.parametrize(
    "changes",
    [
        dict(mode="unknown"),
        dict(mode="raw"),
        dict(target_bytes=512),
        dict(positions=torch.arange(3)),
        dict(positions=torch.arange(512) + 1),
        dict(workspace=object()),
    ],
)
def test_unsupported_scope_fails_before_forward(changes):
    adapter, prepared, workspace, positions = fixture()
    kwargs = dict(mode="source", workspace=workspace, positions=positions)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        StreamingComponentPrefill(adapter, prepared, **kwargs)


def test_once_only_and_no_padded_or_reused_prefill():
    adapter, prepared, workspace, positions = fixture()
    with StreamingComponentPrefill(
        adapter, prepared, mode="source", workspace=workspace, positions=positions
    ) as stream:
        stream.prefill()
        with pytest.raises(RuntimeError, match="only once"):
            stream.prefill()
    prepared.model_inputs["attention_mask"] = torch.tensor([[0, 1, 1, 1, 1, 1, 1]])
    with pytest.raises(ValueError, match="unpadded"):
        StreamingComponentPrefill(
            adapter, prepared, mode="source", workspace=workspace, positions=positions
        )
    del prepared.model_inputs["attention_mask"]
    prepared.model_inputs["past_key_values"] = DynamicCache()
    with pytest.raises(ValueError, match="empty private cache"):
        StreamingComponentPrefill(
            adapter, prepared, mode="source", workspace=workspace, positions=positions
        )
