"""Consumed-input capture and packet feasibility, not an online history codec."""

from contextlib import ExitStack
from typing import Any

import torch


class HistoryProjectionCapture:
    """Read-only normalized-input and *actual cached* post-RoPE K/V capture."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self.stack = ExitStack()
        self.records: dict[int, list[dict[str, Any]]] = {}

    def __enter__(self) -> "HistoryProjectionCapture":
        import transformers

        if transformers.__version__ != "4.55.2":
            raise NotImplementedError("history capture requires pinned Transformers 4.55.2")
        try:
            for index, layer in enumerate(self.adapter._language_layers()):
                module = layer.self_attn
                if (
                    type(module).__name__ != "LlamaAttention"
                    or type(module).__module__ != "transformers.models.llama.modeling_llama"
                ):
                    raise NotImplementedError("audited LlamaAttention only")
                self.records[index] = []

                def before(_module, args, kwargs, *, layer_id=index):
                    hidden = kwargs.get("hidden_states", args[0] if args else None)
                    rotary = kwargs.get("position_embeddings")
                    cache = kwargs.get("past_key_value")
                    positions = kwargs.get("position_ids")
                    if (
                        hidden is None
                        or hidden.ndim != 3
                        or hidden.shape[:2] != (1, 1)
                        or hidden.dtype != torch.bfloat16
                        or not hidden.is_contiguous()
                        or rotary is None
                        or cache is None
                        or positions is None
                        or positions.numel() != 1
                    ):
                        raise ValueError("one-token BF16 source, actual RoPE and cache required")
                    self.records[layer_id].append(
                        dict(
                            normalized=hidden.detach().clone(),
                            cos=rotary[0].detach().clone(),
                            sin=rotary[1].detach().clone(),
                            logical_position=positions.detach().clone(),
                            cache=cache,
                        )
                    )

                def after(_module, args, kwargs, output, *, layer_id=index):
                    del args, kwargs, output
                    record = self.records[layer_id][-1]
                    layer_cache = record.pop("cache").layers[layer_id]
                    used = layer_cache.tail_used
                    if used < 1 or len(layer_cache.history) != 2:
                        raise ValueError("actual immutable raw-tail history entry required")
                    record["key"], record["value"] = tuple(
                        t[..., used - 1 : used, :].detach().clone() for t in layer_cache.history
                    )

                before_handle = module.register_forward_pre_hook(before, with_kwargs=True)
                self.stack.callback(before_handle.remove)
                after_handle = module.register_forward_hook(after, with_kwargs=True)
                self.stack.callback(after_handle.remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.stack.__exit__(*args)

    def assemble(self, updates: int) -> dict[int, dict[str, Any]]:
        if len(self.records) != len(self.adapter._language_layers()):
            raise ValueError("complete layers required")
        result = {}
        for layer, rows in self.records.items():
            if len(rows) != updates or any("key" not in row or "cache" in row for row in rows):
                raise ValueError("complete observed source/cache pairs required")
            positions = torch.cat([row["logical_position"].reshape(-1) for row in rows])
            if not torch.equal(positions[1:], positions[:-1] + 1):
                raise ValueError("original history positions must be consecutive")
            result[layer] = dict(
                normalized=torch.cat([row["normalized"] for row in rows], 1),
                cos=torch.cat([row["cos"] for row in rows], 1),
                sin=torch.cat([row["sin"] for row in rows], 1),
                logical_positions=positions,
                key=torch.cat([row["key"] for row in rows], 2),
                value=torch.cat([row["value"] for row in rows], 2),
            )
        return result


def packet_feasibility(
    adapter: Any, records: Any, *, updates: int, maximum_seconds: float = 300
) -> dict[str, Any]:
    """Charge a hypothetical source-tail codec; no online saving is claimed.

    Existing K/V tail capacity stays sixteen positions. A hypothetical online
    source-capture tail of sixteen normalized inputs is charged in addition.
    Every sealed candidate includes exact original-bit patches and original
    RoPE phases. Original lifetime/constructor peaks remain uncertified.
    """
    import time

    from mosaickv.evaluation.dual_projection import build_dual
    from mosaickv.evaluation.encoded_projection import build
    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.projection_archive import ProjectionArchive

    if type(updates) is not int or not 16 < updates <= 127 or not records:
        raise ValueError("bounded complete 17--127 history updates required")
    began = time.monotonic()
    sealed_count = (updates - 1) // 16
    sealed_positions = sealed_count * 16
    common = records[0]
    cos = common["cos"][:, :sealed_positions].contiguous().clone()
    sin = common["sin"][:, :sealed_positions].contiguous().clone()
    logical = common["logical_positions"][:sealed_positions].contiguous().clone()
    original_packets = []
    source_packets = [cos, sin, logical]
    row_records = []
    raw_tail = source_tail = 0
    for layer, record in records.items():
        attention = adapter._language_layers()[layer].self_attn
        heads, dimension = record["key"].shape[1], record["key"].shape[-1]
        if (
            dimension != 128
            or heads * dimension != record["normalized"].shape[-1]
            or not torch.equal(record["cos"][:, :sealed_positions], cos)
            or not torch.equal(record["sin"][:, :sealed_positions], sin)
            or not torch.equal(record["logical_positions"][:sealed_positions], logical)
        ):
            raise ValueError("same-phase full-MHA history required")
        raw_tail += 2 * heads * 16 * dimension * 2
        source_tail += 16 * heads * dimension * 2
        for chunk in range(sealed_count):
            if time.monotonic() - began > maximum_seconds:
                raise TimeoutError("registered packet-construction diagnostic budget exceeded")
            start, end = chunk * 16, (chunk + 1) * 16
            pair = tuple(record[name][..., start:end, :].contiguous() for name in ("key", "value"))
            for tensor in pair:
                original_packets.extend(encode(tensor).buffers)
            union = torch.arange(16, device=cos.device, dtype=torch.int32)
            selected = union[None].expand(heads, -1)
            archive = ProjectionArchive(
                encode(record["normalized"][:, None, start:end].contiguous()),
                union,
                selected,
                16,
                cos[:, start:end],
                sin[:, start:end],
            )
            base = ExactFormat("source", tuple(pair[0].shape), archive=archive)
            dual, proof = build_dual(base, attention, pair)
            candidate, encoded_proof = build(dual, attention)
            source_packets.extend(candidate.tensors())
            row_records.append(
                dict(
                    layer=layer,
                    chunk=chunk,
                    original_logical_positions=logical[start:end].tolist(),
                    dual=proof,
                    encoded=encoded_proof,
                )
            )
    torch.cuda.synchronize()
    old_sealed = backing_bytes(tuple(original_packets))
    new_sealed = backing_bytes(tuple(source_packets))
    old_total = old_sealed + raw_tail
    new_total = new_sealed + raw_tail + source_tail
    return dict(
        kind="packet_feasibility_not_online_memory_measurement",
        updates=updates,
        sealed_chunks_per_layer=sealed_count,
        existing_sealed_independent_KV_packet_bytes=old_sealed,
        candidate_sealed_source_packet_bytes=new_sealed,
        unchanged_raw_KV_tail_capacity_bytes=raw_tail,
        additional_hypothetical_raw_source_tail_bytes=source_tail,
        existing_history_packet_total_bytes=old_total,
        candidate_history_packet_total_bytes=new_total,
        prospective_packet_saving_fraction=(old_total - new_total) / old_total,
        independent_packet_accounting_not_actual_batched_cache_backing=True,
        original_bits_checked_three_times_per_construction=True,
        original_rope_phases_and_logical_positions_retained=True,
        online_constructor_latency_unmeasured=True,
        whole_request_and_constructor_peak_uncertified=True,
        elapsed_seconds=time.monotonic() - began,
        packets=row_records,
    )
