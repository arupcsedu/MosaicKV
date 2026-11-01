from dataclasses import replace

import pytest
import torch

from mosaickv.evaluation.lossless_affine_minimum import eligible_parent, encode_minimum
from mosaickv.evaluation.precision_tier import pack, unpack_cpu


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("axis", [2, 3])
def test_all_decoded_bits_preserved_without_scale_recalibration(bits, axis):
    torch.manual_seed(103)
    x = torch.randn(1, 2, 136, 128).bfloat16()
    body = pack(x, bits=bits, axis=axis)
    new = encode_minimum(body)
    assert new.codes is body.codes and new.scale is body.scale
    assert new.minimum.element_size() * 2 == body.minimum.element_size()
    assert torch.equal(unpack_cpu(new).view(torch.int16), unpack_cpu(body).view(torch.int16))


def test_unsafe_headers_fail_closed_without_fallback():
    body = pack(torch.randn(1, 1, 16, 128).bfloat16(), bits=3, axis=2)
    bad = body.minimum.clone()
    bad.flatten()[0] = 1.001
    with pytest.raises(ValueError, match="not exactly"):
        encode_minimum(replace(body, minimum=bad))
    with pytest.raises(ValueError):
        encode_minimum(replace(body, scale=body.scale.half()))
    with pytest.raises(ValueError):
        encode_minimum(replace(body, minimum=body.minimum * float("nan")))


def parent():
    summaries = []
    for dataset in ("PrefixKV/LLaVA-Description", "PrefixKV/MM-Vet"):
        summaries.append(
            dict(
                provenance=dict(dataset=dict(dataset_id=dataset)),
                rows=[
                    dict(
                        variant="source3_fp32_tiled_text",
                        PPL=3.0,
                        evaluated_token_count=97,
                        fullkv_reference_ROUGE_L=0.9,
                    ),
                    dict(
                        variant="shadow",
                        PPL=4.0,
                        evaluated_token_count=97,
                        fullkv_reference_ROUGE_L=0.8,
                    ),
                ],
            )
        )
    return dict(complete=True, failures=[], teacher_metric_summary=summaries)


def test_automatic_successor_requires_both_tasks_both_metrics_and_complete_parent():
    assert eligible_parent(parent())
    for field, value in (
        ("PPL", 4.01),
        ("fullkv_reference_ROUGE_L", 0.79),
        ("PPL", float("nan")),
        ("evaluated_token_count", 96),
    ):
        record = parent()
        record["teacher_metric_summary"][1]["rows"][0][field] = value
        assert not eligible_parent(record)
    record = parent()
    record["failures"] = ["failed"]
    assert not eligible_parent(record)
    record = parent()
    record["teacher_metric_summary"] = record["teacher_metric_summary"][:1]
    assert not eligible_parent(record)
    assert not eligible_parent({})
