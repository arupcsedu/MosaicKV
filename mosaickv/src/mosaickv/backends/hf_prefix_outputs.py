"""Read-only single-token Llama hooks for a consumed-input mechanistic replay.

Audited against Transformers 4.55.2 LlamaAttention/LlamaDecoderLayer. These
observations never score a proposal or change the registered stability gate.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


class PrefixOutputRecorder:
    FIELDS = (
        "attention_heads",
        "projected_attention",
        "residual_input",
        "normalized_attention_input",
        "post_attention_residual",
        "layer_output",
    )

    def __init__(self, adapter: Any) -> None:
        import hashlib
        import inspect
        from pathlib import Path

        import transformers
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer

        self.layers = tuple(adapter._language_layers())
        if (
            transformers.__version__ != "4.55.2"
            or not self.layers
            or any(
                type(layer) is not LlamaDecoderLayer
                or type(layer.self_attn) is not LlamaAttention
                or layer.self_attn.config._attn_implementation != "eager"
                for layer in self.layers
            )
        ):
            raise ValueError("prefix output hooks require audited Transformers 4.55.2 eager Llama")
        source = Path(inspect.getfile(LlamaDecoderLayer))
        self.source_contract = dict(
            transformers_version=transformers.__version__,
            source_file=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            classes=["LlamaAttention", "LlamaDecoderLayer"],
            sites=[
                "o_proj.input",
                "o_proj.output",
                "decoder.input",
                "input_layernorm.output",
                "post_attention_layernorm.input",
                "decoder.output",
            ],
        )
        self.records: dict[str, dict[int, Any]] = {}
        self.active = False

    @contextmanager
    def capture(self, role: str, step: int):
        import torch

        if self.active or type(step) is not int or not 1 <= step <= 32:
            raise ValueError("nested or invalid prefix-output capture")
        if step in self.records.get(role, {}):
            raise ValueError("duplicate role/prefix output capture")
        self.active = True
        handles: list[Any] = []
        rows: list[dict[str, Any]] = [dict() for _ in self.layers]

        def save(index, field, value):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or tuple(value.shape[:2]) != (1, 1)
                or not torch.isfinite(value).all()
                or field in rows[index]
            ):
                raise ValueError(
                    "prefix hooks require one finite single-token invocation per layer"
                )
            tensor = value[0, 0].detach().cpu().clone()
            if field == "attention_heads":
                tensor = tensor.reshape(-1, self.layers[index].self_attn.head_dim)
            rows[index][field] = tensor

        def input_hook(index, field):
            def hook(_module, args, kwargs):
                save(
                    index, field, kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                )

            return hook

        def output_hook(index, field):
            def hook(_module, _args, output):
                save(index, field, output)

            return hook

        try:
            for i, layer in enumerate(self.layers):
                for module, field in (
                    (layer, "residual_input"),
                    (layer.self_attn.o_proj, "attention_heads"),
                    (layer.post_attention_layernorm, "post_attention_residual"),
                ):
                    handles.append(
                        module.register_forward_pre_hook(input_hook(i, field), with_kwargs=True)
                    )
                for module, field in (
                    (layer.input_layernorm, "normalized_attention_input"),
                    (layer.self_attn.o_proj, "projected_attention"),
                    (layer, "layer_output"),
                ):
                    handles.append(module.register_forward_hook(output_hook(i, field)))
            yield
            if any(set(row) != set(self.FIELDS) for row in rows):
                raise RuntimeError("incomplete prefix output capture")
            self.records.setdefault(role, {})[step] = tuple(rows)
        finally:
            for handle in handles:
                handle.remove()
            self.active = False

    @property
    def storage_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for steps in self.records.values()
            for layers in steps.values()
            for layer in layers
            for t in layer.values()
        )


def frozen_query_prediction(source, anchor, proposal, evidence, projection):
    """Save the old surrogate's per-prefix vectors and linear projected change.

    This deliberately holds FullKV queries/generated values fixed. The actual
    compressed forwards are captured separately; this is NOT their prediction
    being relabeled as an observation. CPU FP64 projection is sideband only.
    """
    head = proposal.head
    data = evidence[proposal.layer]
    p = data["prompt_probabilities"][head].float()
    values = source.layers[proposal.layer].value[0, head].detach().cpu().float()
    logical = source.logical_positions.physical_to_logical
    physical = {position: i for i, position in enumerate(logical)}
    chosen = sorted(
        physical[row["logical_position"]]
        for row in anchor.slot_records
        if (row["layer"], row["kv_head"]) == (proposal.layer, head)
    )
    numerator = p[:, chosen] @ values[chosen] + data["generated_numerator"][head].float()
    denominator = p[:, chosen].sum(-1) + data["generated_mass"][head].float()
    changed_num = (
        numerator
        + p[:, proposal.added_position, None] * values[proposal.added_position]
        - p[:, proposal.removed_position, None] * values[proposal.removed_position]
    )
    changed_den = denominator + p[:, proposal.added_position] - p[:, proposal.removed_position]
    if not bool((denominator > 0).all() and (changed_den > 0).all()) or p.shape[0] % 32:
        raise ValueError("invalid frozen-query prediction geometry or retained mass")
    groups, dimension = p.shape[0] // 32, values.shape[-1]
    before = (numerator / denominator[:, None]).reshape(32, groups, dimension)
    after = (changed_num / changed_den[:, None]).reshape_as(before)
    target = data["target"][head].float().reshape_as(before)
    start, end = head * groups * dimension, (head + 1) * groups * dimension
    weight = projection.weight[:, start:end].detach().cpu().double().clone()
    projected_change = (after - before).flatten(1).double() @ weight.T
    return dict(
        fullkv_query_target=target,
        predicted_anchor=before,
        predicted_candidate=after,
        predicted_projected_change_fp64=projected_change,
        changed_head_projection_columns_fp64=weight,
        query_heads_per_kv_head=groups,
        numerical_role="sideband_frozen_FullKV_queries_and_generated_values_not_actual_output",
    )
