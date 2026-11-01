"""Request-isolated compact validity transport for padded prompt completion.

The final additive mask is the ONLY allocated CUDA mask workspace and is
charged to decode-owned cache peak. History is valid; padding is never a token.
"""

from contextlib import ExitStack

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def build_mask(counts, length, prompt_length, dtype, old, *, log_mass_bias=None):
    if (
        counts.ndim != 1
        or counts.dtype != torch.int16
        or length < prompt_length
        or not dtype.is_floating_point
        or (
            old is not None
            and (
                old.ndim != 4
                or old.shape[0] != 1
                or old.shape[1] not in (1, counts.numel())
                or old.shape[2] != 1
                or old.shape[-1] < length
                or old.dtype != dtype
                or old.device != counts.device
            )
        )
    ):
        raise ValueError(
            "compatible compact counts, original additive mask and cache shape required"
        )
    if log_mass_bias is not None and (
        log_mass_bias.shape != (counts.numel(), prompt_length)
        or log_mass_bias.dtype != torch.float32
        or log_mass_bias.device != counts.device
        or not log_mass_bias.is_contiguous()
    ):
        raise ValueError("contiguous FP32 head/prompt mass bias on the cache device required")
    mask = torch.empty(1, counts.numel(), 1, length, device=counts.device, dtype=dtype)
    if counts.device.type == "cuda":
        from mosaickv.evaluation.completion_masks_kernel import fill

        fill(mask, counts, prompt_length, old, log_mass_bias)
    else:
        index = torch.arange(length, device=counts.device)
        valid = (index[None] < counts[:, None]) | (index[None] >= prompt_length)
        mask[0, :, 0] = torch.where(valid, 0.0, torch.finfo(dtype).min).to(dtype)
        if old is not None:
            mask += old[..., :length]
        if log_mass_bias is not None:
            work = mask.float()
            work[..., :prompt_length] += log_mass_bias[None, :, None, :]
            mask.copy_(work)
    return mask


def gpu_controls():
    records = []
    for dtype in (torch.bfloat16, torch.float32):
        counts = torch.tensor([3, 5, 4], dtype=torch.int16)
        for causal in (False, True):
            old = torch.zeros(1, 1, 1, 11, dtype=dtype) if causal else None
            if old is not None:
                old[..., 0] = -torch.inf
            expected = build_mask(counts, 9, 5, dtype, old)
            actual = build_mask(counts.cuda(), 9, 5, dtype, None if old is None else old.cuda())
            if not torch.equal(actual.cpu(), expected):
                raise RuntimeError("compact completion mask differs from CPU oracle")
            records.append(dict(dtype=str(dtype), causal_mask=causal, CPU_CUDA_equal=True))
    return records


class CompletionMaskTransport:
    def __init__(self, adapter, cache):
        self.adapter, self.cache = adapter, cache
        self.stack, self.active = ExitStack(), {}

    def restore(self, index):
        if index in self.active:
            mask = self.active.pop(index)
            # Other scoped transports may append persistent source-history
            # owners during attention. Remove only our temporary mask.
            self.cache.shared_tensors = tuple(t for t in self.cache.shared_tensors if t is not mask)
            self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)

    def __enter__(self):
        try:
            for i, block in enumerate(self.adapter._language_layers()):
                layer = self.cache.layers[i]
                fmt = layer.representation
                if not hasattr(fmt, "valid_prompt_counts"):
                    continue
                attention = block.self_attn
                if getattr(attention.config, "_attn_implementation", None) != "eager":
                    raise NotImplementedError("explicit eager completion masks only")

                def pre(_module, args, kwargs, *, index=i, target=layer, packet=fmt):
                    if self.active or kwargs.get("past_key_value") is not self.cache:
                        raise RuntimeError("nested or foreign request in completion transport")
                    hidden = kwargs.get("hidden_states", args[0] if args else None)
                    if hidden is None or hidden.shape[:2] != (1, 1):
                        raise NotImplementedError("batch-one, query-one completion decode required")
                    old = kwargs.get("attention_mask")
                    mask = build_mask(
                        packet.valid_prompt_counts,
                        target.get_seq_length() + 1,
                        packet.shape[-2],
                        hidden.dtype if old is None else old.dtype,
                        old,
                        log_mass_bias=getattr(packet, "log_mass_bias", None),
                    )
                    self.active[index] = mask
                    self.cache.shared_tensors = (*self.cache.shared_tensors, mask)
                    self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
                    self.cache.observe()
                    kwargs["attention_mask"] = mask
                    return args, kwargs

                def post(_module, _args, _output, *, index=i):
                    self.restore(index)

                self.stack.callback(
                    attention.register_forward_pre_hook(pre, with_kwargs=True).remove
                )
                self.stack.callback(attention.register_forward_hook(post, always_call=True).remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args):
        try:
            return self.stack.__exit__(*args)
        finally:
            for index in tuple(self.active):
                self.restore(index)
