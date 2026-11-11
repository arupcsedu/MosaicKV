"""Audited state-aware loading for the combined legacy LLaVA checkpoint.

Only the explicitly verified obsolete rotary buffers may remain unconsumed.
All vision parameters are instantiated before the parent strict weight load.
CPU schema inspection uses mmap and reads only tiny rotary-buffer values.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path

import audit_lookm_hf_generation_compat as generation

VISION_PREFIX = "model.vision_tower.vision_tower."
ROTARY_SOURCE_SHA = "7e50d25e1ab6255866b60ee8a4aa00cbb885dd7eee33c55ab8d4cd8c50446d54"


def checkpoint_compatibility_class(original, vision_directory):
    from transformers import CLIPImageProcessor, CLIPVisionModel

    base = generation.compatibility_class(original)
    vision_directory = Path(vision_directory).resolve(strict=True)

    class CheckpointVisionCompat(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tower = self.get_vision_tower()
            if tower is None or tower.is_loaded or not hasattr(tower, "cfg_only"):
                raise ValueError("Expected the original lazy local CLIP tower")
            # Public constructor, not nested from_pretrained under a parent meta
            # context. Every parameter will be supplied by the combined checkpoint.
            tower.vision_tower = CLIPVisionModel(tower.cfg_only)
            tower.vision_tower.requires_grad_(False)
            tower.image_processor = CLIPImageProcessor.from_pretrained(
                vision_directory,
                local_files_only=True,
            )
            tower.is_loaded = True

    return CheckpointVisionCompat


class MappedCheckpoint:
    def __init__(self, directory, index):
        self.directory, self.index, self.shards = Path(directory), index, {}

    def __enter__(self):
        import torch

        for name in sorted(set(self.index.values())):
            if Path(name).name != name:
                raise ValueError("Unexpected checkpoint shard path")
            # No dense-load fallback if this legacy file is not mmap compatible.
            self.shards[name] = torch.load(
                self.directory / name,
                mmap=True,
                weights_only=True,
                map_location="cpu",
            )
        return self

    def __exit__(self, *_args):
        self.shards.clear()

    def __getitem__(self, key):
        return self.shards[self.index[key]][key]


def verify_schema(
    expected_state,
    index,
    rotary_buffers,
    reader,
    *,
    expected_vision_count=391,
    source_native_proof=None,
):
    """Shapes for all weights; exact values only for the excluded tiny buffers."""
    import torch

    expected, actual = set(expected_state), set(index)
    allowed = set(rotary_buffers)
    if not allowed or allowed & expected or actual - expected != allowed or expected - actual:
        raise ValueError("Checkpoint has missing or unreviewed unexpected keys")
    vision = {key for key in actual if key.startswith(VISION_PREFIX)}
    if len(vision) != expected_vision_count or vision != {
        k for k in expected if k.startswith(VISION_PREFIX)
    }:
        raise ValueError("Every registered vision parameter must be bound to checkpoint state")
    for key, target in expected_state.items():
        if tuple(reader[key].shape) != tuple(target.shape):
            raise ValueError(f"Checkpoint shape mismatch: {key}")
    checked = []
    for key in sorted(allowed):
        saved, reference = reader[key], rotary_buffers[key]
        if (
            saved.device.type != "cpu"
            or reference.device.type != "cpu"
            or saved.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or saved.shape != reference.shape
            or not torch.isfinite(saved).all()
        ):
            raise ValueError(f"Unsupported obsolete rotary buffer: {key}")
        expected_bits = reference.detach().to(saved.dtype).contiguous().view(torch.uint8)
        actual_bits = saved.detach().contiguous().view(torch.uint8)
        equal = torch.equal(actual_bits, expected_bits)
        if not equal and source_native_proof is None:
            raise ValueError(
                f"Obsolete rotary buffer differs from actual configured frequency: {key}"
            )
        if source_native_proof is not None and (
            source_native_proof.get("passed") is not True
            or set(source_native_proof["verified_keys"]) != allowed
        ):
            raise ValueError("Incomplete source-native ignored-buffer proof")
        checked.append(
            {
                "key": key,
                "dtype": str(saved.dtype),
                "shape": list(saved.shape),
                "sha256": hashlib.sha256(actual_bits.numpy().tobytes()).hexdigest(),
                "value_equality_after_declared_cast": equal,
                "value_mismatches": int((saved != reference.to(saved.dtype)).sum()),
                "maximum_absolute_value_difference": float(
                    (saved.float() - reference.to(saved.dtype).float()).abs().max()
                ),
            }
        )
    return {
        "passed": True,
        "expected_consumed_keys": len(expected),
        "vision_keys_bound_to_combined_checkpoint": len(vision),
        "vision_key_set_sha256": hashlib.sha256(json.dumps(sorted(vision)).encode()).hexdigest(),
        "verified_obsolete_rotary_keys": sorted(allowed),
        "rotary_checks": checked,
        "all_other_missing_unexpected_and_shape_mismatches_rejected": True,
        "dense_checkpoint_values_loaded_on_CPU": False,
        "criterion": "strict_obsolete_value_equality"
        if source_native_proof is None
        else "source_native_ignored_serialized_buffers",
        "preserved_obsolete_value_equality_gate": {
            "passed": all(row["value_equality_after_declared_cast"] for row in checked),
            "comparison": "bitwise_after_declared_saved_dtype_cast_not_relaxed",
        },
        "source_native_irrelevance_proof": source_native_proof,
    }


def model_rotary_buffers(model):
    result = {}
    for index, layer in enumerate(model.model.layers):
        rotary = layer.self_attn.rotary_emb
        if (
            rotary.base != model.config.rope_theta
            or rotary.dim != model.config.hidden_size // model.config.num_attention_heads
            or model.config.rope_scaling is not None
        ):
            raise ValueError("Unreviewed original rotary configuration")
        result[f"model.layers.{index}.self_attn.rotary_emb.inv_freq"] = rotary.inv_freq
    return result


def prove_unconsumed_rotary_module(module, saved, *, required_length):
    """Demonstrate ignored loading and unchanged buffers/phases, not value equality."""
    import torch

    names = ("inv_freq", "cos_cached", "sin_cached")
    if (
        not set(names).issubset(module._non_persistent_buffers_set)
        or "inv_freq" in module.state_dict()
        or not 0 < required_length < module.max_seq_len_cached
    ):
        raise ValueError(
            "Nonpersistent original rotary buffers and a strict prebuilt bound required"
        )
    tensors = tuple(getattr(module, name) for name in names)
    before = tuple(t.detach().clone() for t in tensors)
    pointers = tuple(t.data_ptr() for t in tensors)
    cached = module.max_seq_len_cached
    x = torch.empty((1, 1, 1, module.dim), dtype=torch.float16, device=tensors[0].device)
    phase_before = tuple(t.clone() for t in module(x, seq_len=required_length))
    loaded = module.load_state_dict({"inv_freq": saved}, strict=False)
    phase_after = module(x, seq_len=required_length)
    after = tuple(getattr(module, name) for name in names)
    if (
        loaded.missing_keys
        or loaded.unexpected_keys != ["inv_freq"]
        or pointers != tuple(t.data_ptr() for t in after)
        or cached != module.max_seq_len_cached
        or any(
            not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(before, after, strict=True)
        )
        or any(
            not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(phase_before, phase_after, strict=True)
        )
    ):
        raise ValueError("Serialized rotary value affected actual module buffers or forward phases")
    return {
        "passed": True,
        "serialized_key_is_unexpected": True,
        "all_buffer_bits_and_pointers_unchanged": True,
        "forward_phase_bits_unchanged": True,
        "required_position_length": required_length,
        "prebuilt_position_length": cached,
        "rope_rewritten": False,
    }


def source_native_rotary_proof(model, reader, *, required_length):
    """Only the pinned per-attention rotary implementation is admitted."""
    checks = []
    for key in model_rotary_buffers(model):
        index = int(key.split(".")[2])
        rotary = model.model.layers[index].self_attn.rotary_emb
        source = Path(inspect.getsourcefile(type(rotary)))
        if (
            type(rotary).__name__ != "LlamaRotaryEmbedding"
            or type(rotary).__module__ != "llava.model.kv_token_merge.v433_modeling_llama"
            or hashlib.sha256(source.read_bytes()).hexdigest() != ROTARY_SOURCE_SHA
        ):
            raise ValueError("Unreviewed rotary source; unused-key semantics cannot be inferred")
        checks.append(
            {
                "key": key,
                **prove_unconsumed_rotary_module(
                    rotary,
                    reader[key],
                    required_length=required_length,
                ),
            }
        )
    if len(checks) != 32:
        raise ValueError("All 32 actual per-attention rotary modules must be verified")
    return {
        "passed": True,
        "verified_keys": [row["key"] for row in checks],
        "checks": checks,
        "source_sha256": ROTARY_SOURCE_SHA,
        "required_position_length": required_length,
        "prebuilt_position_length": checks[0]["prebuilt_position_length"],
        "all_serialized_rotary_values_ignored_without_rewrite": True,
    }


def meta_audit(original, model_directory, vision_directory, index, *, required_length):
    from accelerate import init_empty_weights

    config = original.config_class.from_pretrained(model_directory, local_files_only=True)
    config.mm_vision_tower = str(vision_directory)
    cls = checkpoint_compatibility_class(original, vision_directory)
    # Keep only small real buffers (frequencies/phase tables); every learned
    # parameter, including CLIP, is meta. No dense model weights are allocated.
    with init_empty_weights(include_buffers=False):
        model = cls(config, kv_mode="origin", hh_ratio=0.1, recent_ratio=0.1)
    if any(not parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Meta schema construction allocated a learned parameter")
    with MappedCheckpoint(model_directory, index) as reader:
        proof = source_native_rotary_proof(model, reader, required_length=required_length)
        report = verify_schema(
            model.state_dict(),
            index,
            model_rotary_buffers(model),
            reader,
            source_native_proof=proof,
        )
    return report


def verify_loading_info(loading, schema):
    if (
        loading.get("missing_keys")
        or loading.get("mismatched_keys")
        or loading.get("error_msgs")
        or set(loading.get("unexpected_keys", ())) - set(schema["verified_obsolete_rotary_keys"])
    ):
        raise ValueError("Strict checkpoint loading reported an unreviewed discrepancy")


def verify_vision_values(state, reader, schema, *, chunk_elements=262144):
    """Bounded CPU-transfer chunks verify every loaded vision word on the worker."""
    import torch

    keys = sorted(key for key in state if key.startswith(VISION_PREFIX))
    if (
        len(keys) != schema["vision_keys_bound_to_combined_checkpoint"]
        or hashlib.sha256(json.dumps(keys).encode()).hexdigest() != schema["vision_key_set_sha256"]
    ):
        raise ValueError("Loaded vision key set differs from the verified checkpoint schema")
    scalars = 0
    for key in keys:
        actual, reference = state[key].detach(), reader[key]
        if actual.is_meta or not actual.is_contiguous() or not reference.is_contiguous():
            raise ValueError("Vision parameter is unmaterialized or has an unreviewed layout")
        if actual.shape != reference.shape:
            raise ValueError("Loaded vision shape differs from checkpoint")
        for start in range(0, actual.numel(), chunk_elements):
            chunk = actual.view(-1)[start : start + chunk_elements].cpu().contiguous()
            expected = (
                reference.view(-1)[start : start + chunk_elements].to(chunk.dtype).contiguous()
            )
            if not torch.equal(chunk.view(torch.uint8), expected.view(torch.uint8)):
                raise ValueError(f"Loaded vision value differs from combined checkpoint: {key}")
            scalars += chunk.numel()
    return {
        "all_vision_parameter_values_verified": True,
        "keys": len(keys),
        "scalars": scalars,
        "maximum_comparison_chunk_elements": chunk_elements,
        "effective_vision_weights": "registered_combined_checkpoint_not_external_reload",
    }


@contextmanager
def rotary_runtime_guard(model, schema):
    """Reject cached-phase rebuilds or positions outside the source-proven bound."""
    proof = schema["source_native_irrelevance_proof"]
    required, prebuilt = proof["required_position_length"], proof["prebuilt_position_length"]
    if proof["passed"] is not True or not 0 < required < prebuilt:
        raise ValueError("Source-proven bounded rotary execution required")
    handles, calls, maximum = [], [0] * len(model.model.layers), [0] * len(model.model.layers)

    def fingerprint(rotary):
        return (
            rotary.max_seq_len_cached,
            tuple(
                (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
                for t in (rotary.inv_freq, rotary.cos_cached, rotary.sin_cached)
            ),
        )

    try:
        for index, layer in enumerate(model.model.layers):
            rotary = layer.self_attn.rotary_emb
            original = fingerprint(rotary)
            if original[0] != prebuilt:
                raise ValueError("Unexpected original rotary cache capacity")

            def before(module, args, kwargs, *, slot=index, expected=original):
                length = kwargs.get("seq_len", args[1] if len(args) > 1 else None)
                if (
                    type(length) is not int
                    or not 0 < length <= required
                    or length >= prebuilt
                    or fingerprint(module) != expected
                ):
                    raise ValueError(
                        "Rotary position exceeded registered prebuilt range or cache changed"
                    )
                calls[slot] += 1
                maximum[slot] = max(maximum[slot], length)

            def after(module, _args, _kwargs, _output, *, expected=original):
                if fingerprint(module) != expected:
                    raise ValueError("Unexpected rotary cache rebuild")

            handles.append(rotary.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(rotary.register_forward_hook(after, with_kwargs=True))
        yield {
            "calls_per_layer": calls,
            "maximum_length_per_layer": maximum,
            "registered_required_length": required,
            "prebuilt_length": prebuilt,
        }
    finally:
        for handle in handles:
            handle.remove()
