"""Strict combined-checkpoint schema; no large weights or GPU execution."""

from types import SimpleNamespace

import lookm_checkpoint_schema as study
import pytest
import torch


def fixture():
    expected = {
        study.VISION_PREFIX + "one.weight": torch.empty(2, device="meta"),
        study.VISION_PREFIX + "two.weight": torch.empty(3, device="meta"),
        "model.core.weight": torch.empty((2, 2), device="meta"),
    }
    rotary = {
        f"model.layers.{i}.self_attn.rotary_emb.inv_freq": torch.tensor([1.0, 0.1])
        for i in range(2)
    }
    values = {key: torch.zeros(target.shape) for key, target in expected.items()}
    values.update({key: value.half() for key, value in rotary.items()})
    index = dict.fromkeys(values, "shard.bin")
    return expected, index, rotary, values


def test_lazy_vision_omission_cannot_be_whitelisted():
    expected, index, rotary, values = fixture()
    del expected[study.VISION_PREFIX + "one.weight"]
    with pytest.raises(ValueError, match="unexpected keys"):
        study.verify_schema(expected, index, rotary, values, expected_vision_count=2)


def test_every_vision_parameter_and_exact_rotary_value_is_verified():
    report = study.verify_schema(*fixture(), expected_vision_count=2)
    assert report["vision_keys_bound_to_combined_checkpoint"] == 2
    assert len(report["rotary_checks"]) == 2
    assert report["dense_checkpoint_values_loaded_on_CPU"] is False
    study.verify_loading_info({"unexpected_keys": report["verified_obsolete_rotary_keys"]}, report)


def test_wrong_rotary_value_is_not_ignored():
    expected, index, rotary, values = fixture()
    values[next(iter(rotary))][0] = 2
    with pytest.raises(ValueError, match="configured frequency"):
        study.verify_schema(expected, index, rotary, values, expected_vision_count=2)


def test_arbitrary_extra_key_or_shape_mismatch_fails():
    expected, index, rotary, values = fixture()
    index["unexpected.weight"] = "shard.bin"
    with pytest.raises(ValueError, match="unexpected keys"):
        study.verify_schema(expected, index, rotary, values, expected_vision_count=2)
    del index["unexpected.weight"]
    values["model.core.weight"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        study.verify_schema(expected, index, rotary, values, expected_vision_count=2)


def test_loading_info_does_not_ignore_vision_or_foreign_discrepancy():
    for info in (
        {"unexpected_keys": [study.VISION_PREFIX + "one.weight"]},
        {"missing_keys": [study.VISION_PREFIX + "one.weight"]},
        {"mismatched_keys": [("model.core.weight", (2, 2), (3, 3))]},
    ):
        with pytest.raises(ValueError, match="unreviewed discrepancy"):
            study.verify_loading_info(info, {"verified_obsolete_rotary_keys": []})


def test_mmap_reads_only_registered_shard_without_dense_fallback(tmp_path):
    _, index, _, values = fixture()
    torch.save(values, tmp_path / "shard.bin")
    with study.MappedCheckpoint(tmp_path, index) as reader:
        assert reader[next(iter(index))].shape == values[next(iter(index))].shape
    assert reader.shards == {}


def test_every_loaded_vision_word_is_verified_in_bounded_chunks():
    expected, index, rotary, values = fixture()
    report = study.verify_schema(expected, index, rotary, values, expected_vision_count=2)
    state = {key: value.half() for key, value in values.items() if key in expected}
    assert study.verify_vision_values(state, values, report, chunk_elements=1)["keys"] == 2
    state[study.VISION_PREFIX + "two.weight"][1] = 1
    with pytest.raises(ValueError, match="value differs"):
        study.verify_vision_values(state, values, report, chunk_elements=1)


def test_rotary_configuration_is_not_guessed():
    model = SimpleNamespace(
        config=SimpleNamespace(
            rope_theta=10000.0, hidden_size=8, num_attention_heads=2, rope_scaling=None
        ),
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        rotary_emb=SimpleNamespace(base=5000.0, dim=4, inv_freq=torch.ones(2))
                    )
                )
            ]
        ),
    )
    with pytest.raises(ValueError, match="configuration"):
        study.model_rotary_buffers(model)


class RotaryFixture(torch.nn.Module):
    def __init__(self, *, persistent=False, mutate=False):
        super().__init__()
        self.dim, self.max_seq_len_cached, self.mutate = 4, 8, mutate
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.1]), persistent=persistent)
        self.register_buffer("cos_cached", torch.ones((1, 1, 8, 4)), persistent=persistent)
        self.register_buffer("sin_cached", torch.zeros((1, 1, 8, 4)), persistent=persistent)

    def forward(self, x, *, seq_len):
        if self.mutate:
            self.cos_cached = self.cos_cached.clone()
        return (
            self.cos_cached[:, :, :seq_len].to(x.dtype),
            self.sin_cached[:, :, :seq_len].to(x.dtype),
        )


def test_source_native_ignore_proof_preserves_failed_value_equivalence():
    rotary = RotaryFixture()
    saved = torch.tensor([9.0, 2.0], dtype=torch.bfloat16)
    proof = study.prove_unconsumed_rotary_module(rotary, saved, required_length=4)
    assert proof["passed"] and proof["forward_phase_bits_unchanged"]
    expected, index, buffers, values = fixture()
    values[next(iter(buffers))] = saved
    with pytest.raises(ValueError, match="configured frequency"):
        study.verify_schema(expected, index, buffers, values, expected_vision_count=2)
    reviewed = {"passed": True, "verified_keys": sorted(buffers)}
    result = study.verify_schema(
        expected, index, buffers, values, expected_vision_count=2, source_native_proof=reviewed
    )
    assert result["passed"]
    assert result["criterion"] == "source_native_ignored_serialized_buffers"
    assert result["preserved_obsolete_value_equality_gate"]["passed"] is False


def test_persistent_or_unbounded_rotary_cannot_use_unused_key_proof():
    with pytest.raises(ValueError, match="Nonpersistent"):
        study.prove_unconsumed_rotary_module(
            RotaryFixture(persistent=True), torch.ones(2), required_length=4
        )
    with pytest.raises(ValueError, match="prebuilt bound"):
        study.prove_unconsumed_rotary_module(RotaryFixture(), torch.ones(2), required_length=8)


def test_runtime_guard_rejects_position_growth_or_cache_rebuild():
    rotary = RotaryFixture()
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary))]
        )
    )
    schema = {
        "source_native_irrelevance_proof": {
            "passed": True,
            "required_position_length": 4,
            "prebuilt_position_length": 8,
        }
    }
    with study.rotary_runtime_guard(model, schema) as record:
        rotary(torch.ones(1), seq_len=3)
    assert record["calls_per_layer"] == [1]
    with pytest.raises(ValueError, match="registered"), study.rotary_runtime_guard(model, schema):
        rotary(torch.ones(1), seq_len=5)
    rotary.mutate = True
    with pytest.raises(ValueError, match="rebuild"), study.rotary_runtime_guard(model, schema):
        rotary(torch.ones(1), seq_len=3)
    assert not rotary._forward_hooks and not rotary._forward_pre_hooks
