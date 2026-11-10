from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import evicastkv_source_protection_diagnostic as diagnostic
import pytest
import torch


def test_real_tensor_hash_binds_values_dtype_and_shape():
    value = torch.arange(8).to(torch.bfloat16)
    original = diagnostic.tensor_hash(value)
    assert original == diagnostic.tensor_hash(value.clone())
    assert original["sha256"] != diagnostic.tensor_hash(value.reshape(2, 4))["sha256"]
    assert original["sha256"] != diagnostic.tensor_hash(value.float())["sha256"]
    value[0] += 1
    assert original["sha256"] != diagnostic.tensor_hash(value)["sha256"]
    with pytest.raises(ValueError, match="nonfinite"):
        diagnostic.tensor_hash(torch.tensor([float("nan")]))


def test_capture_limit_deduplication_and_original_dtype():
    capture = diagnostic.Captures(maximum=8)
    value = torch.zeros((1, 1, 4), dtype=torch.bfloat16)
    capture.save("one", value, vector=True)
    assert capture.bytes == 8
    assert capture.values["one"].dtype == torch.bfloat16
    with pytest.raises(ValueError, match="duplicate"):
        capture.save("one", value, vector=True)
    with pytest.raises(ValueError, match="limit"):
        capture.save("two", value, vector=True)
    with pytest.raises(ValueError, match="BF16"):
        diagnostic.Captures().save("bad", value.float(), vector=True)


class Attention(torch.nn.Module):
    def forward(self, hidden):
        return hidden * 0.5, None


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.post_attention_layernorm = torch.nn.Identity()

    def forward(self, hidden_states):
        hidden_states = self.post_attention_layernorm(
            hidden_states + self.self_attn(hidden_states)[0]
        )
        return hidden_states + 1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.norm = torch.nn.Identity()

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return self.norm(value)


def adapter():
    model = Model()
    value = torch.ones((1, 1, 3), dtype=torch.bfloat16)
    return SimpleNamespace(
        model=model,
        _language_layers=lambda: tuple(model.layers),
        prefill=lambda: model(value),
        decode_one_token=lambda: model(value),
    )


def test_hooks_are_prefix_limited_numerically_transparent_and_removed():
    runtime = adapter()
    original = runtime.decode_one_token
    expected = original().clone()
    capture = diagnostic.Captures()
    with diagnostic.capture_operators(runtime, "arm", capture) as state:
        runtime.prefill()
        for _ in range(127):
            assert torch.equal(runtime.decode_one_token(), expected)
    assert state["updates"] == 127
    assert len(capture.values) == 4 * (2 * 4 + 1)
    assert {int(key.split("|")[1]) for key in capture.values} == {20, 21, 22, 23}
    assert runtime.decode_one_token is original
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in runtime.model.modules()
    )


def test_hooks_rollback_after_exception_and_incomplete_horizon_fails():
    runtime = adapter()
    original = runtime.prefill
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        diagnostic.capture_operators(runtime, "a", diagnostic.Captures()),
    ):
        raise RuntimeError("interrupted")
    assert runtime.prefill is original
    assert all(not module._forward_hooks for module in runtime.model.modules())
    with (
        pytest.raises(ValueError, match="127"),
        diagnostic.capture_operators(runtime, "a", diagnostic.Captures()),
    ):
        runtime.decode_one_token()


def packet_fixture(*, corrupt=False, malformed=False):
    original = tuple(
        torch.arange(2 * 4 * 128).reshape(1, 2, 4, 128).to(torch.bfloat16) for _ in range(2)
    )
    rows = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
    if malformed:
        rows[1] = 0
    restored = tuple(value.clone() for value in original)
    if corrupt:
        restored[0][0, 0, 0, 0] += 1
    packet = SimpleNamespace(
        kind="evicastkv_full_prompt_shared_source_precision_variant",
        source=SimpleNamespace(
            codes=torch.arange(6, dtype=torch.uint8),
            minimum=torch.zeros(1, 2),
            scale=torch.ones(1, 2),
            bits=3,
            axis=2,
            group=128,
        ),
        row_positions=rows,
        cos=torch.ones(1, 4, 128, dtype=torch.bfloat16),
        sin=torch.zeros(1, 4, 128, dtype=torch.bfloat16),
        union=torch.arange(4),
        slots=torch.arange(4).expand(2, 4),
        transport_patches=(),
        materialize=lambda attention: restored,
    )
    runtime = SimpleNamespace(
        _legacy_layers=lambda cache: ((original,), "fixture"),
        _language_layers=lambda: (SimpleNamespace(self_attn=object()),),
    )
    prefill = SimpleNamespace(state=SimpleNamespace(past_key_values=object()))
    return runtime, prefill, packet


def test_chosen_packet_protected_rows_are_bitwise_original_and_workspace_charged():
    runtime, prefill, packet = packet_fixture()
    capture = diagnostic.Captures()
    result = diagnostic.inspect_packets(runtime, prefill, (packet,), "arm", capture)
    assert len(result) == 1
    row = result[0]
    assert row["protected_original_K_bitwise_equal"] and row["protected_original_V_bitwise_equal"]
    assert row["explicit_extra_audit_workspace_bytes"] > 0
    assert not row["audit_scratch_KV_only_peak_certified"]
    assert set(row["tensor_hashes"]) == {
        "codes",
        "minimum",
        "scale",
        "cos",
        "sin",
        "union",
        "slots",
    }
    assert torch.equal(capture.values["arm|packet_rows|layer0"], packet.row_positions)


@pytest.mark.parametrize("malformed,corrupt", [(True, False), (False, True)])
def test_malformed_rows_and_corrupted_exact_overlay_fail(malformed, corrupt):
    runtime, prefill, packet = packet_fixture(corrupt=corrupt, malformed=malformed)
    with pytest.raises((ValueError, RuntimeError)):
        diagnostic.inspect_packets(runtime, prefill, (packet,), "arm", diagnostic.Captures())


def test_inspection_wraps_only_returned_chosen_format_and_restores_builder(monkeypatch):
    import evicastkv_review_source_prefix as source

    chosen = (object(),)

    def original(*args, **kwargs):
        return chosen, ["proof"]

    monkeypatch.setattr(source, "_formats_with_ceiling", original)
    calls = []
    monkeypatch.setattr(
        diagnostic, "inspect_packets", lambda *args: calls.append(args) or [dict(chosen=True)]
    )
    records = []
    with diagnostic.inspect_chosen_packets("arm", diagnostic.Captures(), records):
        assert source._formats_with_ceiling("adapter", "prepared", "prefill") == (chosen, ["proof"])
    assert len(calls) == 1 and calls[0][2] is chosen
    assert source._formats_with_ceiling is original
    assert records == [dict(chosen=True)]


def test_diagnostic_never_changes_registered_source_masks_or_guard():
    import evicastkv_review_source_prefix as source

    assert diagnostic.POLICY["guard"] == source.POLICY["guard"]
    assert (
        diagnostic.POLICY["protection_fractions"] == source.POLICY["optional_protection_fractions"]
    )
    assert diagnostic.POLICY["no_vocabulary_patch"]
    assert diagnostic.POLICY["no_policy_change"]
    text = inspect.getsource(diagnostic.inspect_chosen_packets)
    assert "prefix" not in text.replace("evicastkv_review_source_prefix", "source")
    assert "label" not in text


def test_bound_file_drift_fails_closed(tmp_path, monkeypatch):
    watched = tmp_path / "immutable"
    watched.write_text("a")
    protocol = dict(
        source_sha="abc",
        policy=json.loads(json.dumps(diagnostic.POLICY)),
        arms=diagnostic.ARMS,
        bindings={str(watched): diagnostic.sha(watched)},
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(diagnostic, "clean_source", lambda *args: "abc")
    diagnostic.validate(tmp_path)
    watched.write_text("b")
    with pytest.raises(ValueError, match="binding"):
        diagnostic.validate(tmp_path)


def isolated_snapshot():
    return dict(
        query_error=None,
        background_processes=[],
        visible_process_concurrency=1,
        devices=[
            dict(name="NVIDIA A100-SXM4-80GB", visible_to_process=True),
            dict(name="NVIDIA A100-SXM4-80GB", visible_to_process=False),
        ],
    )


def test_isolation_uses_one_saved_snapshot_without_changing_other_gpu_occupants():
    gpu = isolated_snapshot()
    runtime = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1, get_device_name=lambda: "NVIDIA A100-SXM4-80GB"
        )
    )
    result = diagnostic.require_isolated_slurm_gpu(
        runtime, snapshot=gpu, environ=dict(SLURM_JOB_ID="123", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    )
    assert result is gpu


@pytest.mark.parametrize(
    "issue",
    ["no_slurm", "invalid_slurm", "two_visible", "query_error", "background", "concurrency"],
)
def test_gpu_isolation_fails_closed(issue):
    gpu = isolated_snapshot()
    environment = dict(SLURM_JOB_ID="123", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    runtime = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1, get_device_name=lambda: "NVIDIA A100-SXM4-80GB"
        )
    )
    if issue == "no_slurm":
        environment.pop("SLURM_JOB_ID")
    elif issue == "invalid_slurm":
        environment["SLURM_JOB_ID"] = "123;other"
    elif issue == "two_visible":
        gpu["devices"][1]["visible_to_process"] = True
    elif issue == "query_error":
        gpu["query_error"] = "process query failed"
    elif issue == "background":
        gpu["background_processes"] = [dict(pid=55)]
    else:
        gpu["visible_process_concurrency"] = 2
    with pytest.raises(ValueError):
        diagnostic.require_isolated_slurm_gpu(runtime, snapshot=gpu, environ=environment)
