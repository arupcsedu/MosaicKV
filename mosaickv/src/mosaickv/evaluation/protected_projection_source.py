"""CPU metadata/protection for a numerical source-codec study, not GPU inference."""

import hashlib

import torch


def metadata_adapter(checkpoint):
    """Real adapter preprocessing with meta-device model geometry, no weight load."""
    from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

    from mosaickv.adapters.huggingface.llava import Llava15Adapter

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config).to(dtype=torch.bfloat16)
    if not all(p.device.type == "meta" for p in model.parameters()):
        raise RuntimeError("metadata construction must not allocate real checkpoint weights")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    return Llava15Adapter(model, processor, device=torch.device("cpu"))


def original_text_positions(adapter, messages, *, expected_length):
    prepared = adapter.prepare_inputs(messages)
    if prepared.logical_sequence_length != expected_length:
        raise ValueError("processor route does not match captured source length")
    if not prepared.model_inputs["attention_mask"].bool().all():
        raise ValueError("unpadded authoritative token route required")
    coverage, text = [], []
    for span in prepared.modality_map:
        ids = list(range(span.start, span.end))
        coverage.extend(ids)
        if getattr(span.modality, "value", span.modality) == "text":
            text.extend(ids)
    if coverage != list(range(expected_length)) or expected_length - 1 not in text:
        raise ValueError("complete disjoint modality spans and text generation delimiter required")
    digest = hashlib.sha256(
        prepared.model_inputs["input_ids"].contiguous().numpy().tobytes()
    ).hexdigest()
    return tuple(text), digest


def protected_positions(memberships, text, *, length):
    if not memberships or any(
        not row or tuple(sorted(set(row))) != tuple(row) or row[-1] != length - 1 or row[0] < 0
        for row in memberships
    ):
        raise ValueError("sorted source memberships preserving last token required")
    mask = torch.zeros(len(memberships), length, dtype=torch.bool)
    for h, row in enumerate(memberships):
        exact = set(row[:4]) | set(row[-8:]) | (set(row) & set(text))
        mask[h, sorted(exact)] = True
    return mask
