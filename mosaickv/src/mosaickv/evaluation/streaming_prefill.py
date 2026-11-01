"""Private one-request prefill retirement; never decode through this cache.

The attention module consumes full original K/V before its post-hook serializes
and retires that layer. No offload, pruning, position change or model patch.
Encoders are explicit; only tested encoders may be used in a GPU protocol.
"""

import weakref
from contextlib import ExitStack

import torch
from transformers.cache_utils import Cache, DynamicLayer

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.projection_cache import ScratchAudit, identity


class RetiringLayer(DynamicLayer):
    def __init__(self, **kwargs):
        super().__init__()
        self.owner, self.index, self.length, self.packet = None, None, 0, None

    def update(self, key_states, value_states, cache_kwargs=None):
        if (
            self.owner is None
            or self.index != self.owner.next_layer
            or self.length
            or self.keys is not None
            or self.values is not None
            or key_states.shape != value_states.shape
            or key_states.ndim != 4
            or key_states.shape[0] != 1
            or key_states.shape[-1] != 128
            or key_states.dtype != torch.bfloat16
            or value_states.dtype != torch.bfloat16
            or key_states.device != value_states.device
        ):
            raise ValueError("one original ordered batch-one BF16/MHA prefill transaction required")
        positions = None if cache_kwargs is None else cache_kwargs.get("cache_position")
        if positions is not None and not torch.equal(
            positions, torch.arange(key_states.shape[-2], device=positions.device)
        ):
            raise ValueError("original contiguous prefill positions required")
        self.keys, self.values = key_states, value_states
        for tensor in (key_states, value_states):
            seen = set()
            while tensor is not None and id(tensor) not in seen:
                seen.add(id(tensor))
                self.owner.original_refs.append(weakref.ref(tensor))
                tensor = tensor._base
        self.length = key_states.shape[-2]
        self.owner.observe()
        return self.keys, self.values

    def get_seq_length(self, cache_position=None):
        return self.length

    def crop(self, *args, **kwargs):
        raise NotImplementedError("retiring prefill cache cannot crop or decode")

    def reorder_cache(self, *args, **kwargs):
        raise NotImplementedError("retiring prefill cache supports no beam reordering")


class RetiringPrefillCache(Cache):
    def __init__(self, layer_count, *, audit=True):
        if type(layer_count) is not int or layer_count < 1 or type(audit) is not bool:
            raise ValueError("positive layer count and explicit audit flag required")
        super().__init__(layer_classes=RetiringLayer)
        self.layers = [RetiringLayer() for _ in range(layer_count)]
        self.num_hidden_layers = layer_count
        self.next_layer, self.peak_bytes, self.audit, self.observations = 0, 0, audit, 0
        self.original_refs, self.extra_tensors = [], ()
        for index, layer in enumerate(self.layers):
            layer.owner, layer.index = self, index

    def owned_tensors(self):
        return (
            *self.extra_tensors,
            *tuple(
                tensor
                for layer in self.layers
                for tensor in (
                    *(() if layer.keys is None else (layer.keys, layer.values)),
                    *(() if layer.packet is None else layer.packet.tensors()),
                )
            ),
        )

    def observe(self, extra=()):
        self.peak_bytes = max(self.peak_bytes, backing_bytes((*self.owned_tensors(), *extra)))
        self.observations += 1

    def seal(self, index, attention, encoder, args=(), kwargs=None, output=None):
        if index != self.next_layer or index >= len(self.layers):
            raise ValueError("prefill retirement must follow original layer order")
        layer = self.layers[index]
        if layer.keys is None or layer.packet is not None:
            raise RuntimeError("original unsealed K/V required")
        original = (layer.keys, layer.values)
        excluded = {identity(t) for t in self.owned_tensors()}
        excluded.update(identity(t) for t in attention.parameters())
        from contextlib import nullcontext

        mode = ScratchAudit(excluded=excluded, observe=self.observe) if self.audit else None
        with mode if mode is not None else nullcontext():
            packet = encoder(index, attention, original, args, kwargs or {}, output)
            original_owners = {identity(t) for t in original}
            if any(identity(t) in original_owners for t in packet.tensors()):
                raise RuntimeError("encoded prompt aliases retired original K/V backing")
            if tuple(packet.shape) != tuple(original[0].shape):
                raise RuntimeError("streamed encoder changed physical prompt geometry")
            self.observe((*original, *packet.tensors()))
        layer.packet = packet
        layer.keys = layer.values = None
        self.next_layer += 1
        self.observe(original)

    def formats(self):
        if self.next_layer != len(self.layers) or any(
            x.keys is not None or x.values is not None or x.packet is None for x in self.layers
        ):
            raise RuntimeError("all original layers must be sealed and retired")
        return tuple(x.packet for x in self.layers)

    def record(self, *, require_complete=True):
        if require_complete:
            self.formats()
        return dict(
            boundary="prefill_owned_KV_encoded_payload_and_observed_encoder_overlap",
            final_prefill_cache_bytes=backing_bytes(self.owned_tensors()),
            observed_prefill_cache_peak_bytes=self.peak_bytes,
            original_native_layers_remaining=0,
            logical_lengths=[x.length for x in self.layers],
            observations=self.observations,
            tensor_results_observed=self.audit,
            whole_request_gpu_peak_certified=False,
            CPU_offload_or_residual_bytes=0,
            original_KV_tensor_objects_alive=sum(r() is not None for r in self.original_refs),
            excludes="weights,media_encoder,model_activations_outside_encoder,opaque_GPU_workspace",
        )


class SealAfterAttention:
    def __init__(self, adapter, cache, encoder):
        self.adapter, self.cache, self.encoder, self.stack = adapter, cache, encoder, ExitStack()

    def __enter__(self):
        import transformers

        if transformers.__version__ != "4.55.2":
            raise NotImplementedError("audited Transformers 4.55.2 only")
        try:
            for i, block in enumerate(self.adapter._language_layers()):
                module = block.self_attn
                if (
                    type(module).__name__ != "LlamaAttention"
                    or type(module).__module__ != "transformers.models.llama.modeling_llama"
                    or module.training
                    or module.num_key_value_groups != 1
                    or module.config._attn_implementation != "eager"
                ):
                    raise NotImplementedError("private eval/eager/full-MHA Llama attention only")

                def after(attention, args, kwargs, output, *, index=i):
                    if torch.is_grad_enabled():
                        raise RuntimeError("streamed retirement is inference-only")
                    self.cache.seal(index, attention, self.encoder, args, kwargs, output)

                handle = module.register_forward_hook(after, with_kwargs=True)
                self.stack.callback(handle.remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def lossless_kv_encoder(index, attention, pair, args, kwargs, output):
    from mosaickv.evaluation.exact_storage import ExactFormat, decoded, encode

    packet = ExactFormat("kv", tuple(pair[0].shape), pair=tuple(encode(t) for t in pair))
    for _ in range(3):
        if not all(
            torch.equal(original.view(torch.int16), decoded(body).view(torch.int16))
            for original, body in zip(pair, packet.pair, strict=True)
        ):
            raise RuntimeError("prefill encoding changed an original K/V bit")
    return packet
