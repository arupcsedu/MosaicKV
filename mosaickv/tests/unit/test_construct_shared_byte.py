"""Synthetic construction checks, never measured-model evidence."""

from dataclasses import replace
from types import SimpleNamespace

import evicastkv_construct_shared_byte as study
import pytest
import torch

from mosaickv.evaluation.exact_storage import encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.shared_source_format import SharedSourceFormat


def packet(value):
    return SharedSourceFormat(
        source=pack(torch.full((1, 1, 2, 128), value, dtype=torch.bfloat16), bits=3, axis=3),
        shape=(1, 1, 2, 2),
        row_positions=torch.tensor([0], dtype=torch.int32),
        exact=(encode(torch.full((1, 1, 1, 2), value, dtype=torch.bfloat16)),) * 2,
        cos=torch.ones(1, 2, 2, dtype=torch.bfloat16),
        sin=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        union=torch.arange(2, dtype=torch.int32),
        slots=torch.arange(2, dtype=torch.int16)[None],
    )


def families():
    return {
        key: tuple(packet(1.0 + j + i) for i in range(2)) for j, key in enumerate(study.CHOICES)
    }


def test_exact_constructed_sum_with_verified_alias_sharing_and_no_value_change():
    source = families()
    original = [t.clone() for p in source[study.CHOICES[1]] for t in p.tensors()]
    canonical = study.coalesce_headers(source)
    chosen = (canonical[study.CHOICES[0]][0], canonical[study.CHOICES[1]][1])
    actual = backing_bytes(tuple(t for p in chosen for t in p.tensors()))
    result = study.choose_constructed(source, [study.CHOICES[0], study.CHOICES[1]], actual)
    assert backing_bytes(tuple(t for p in result for t in p.tensors())) == actual
    assert result[0].cos is result[1].cos
    for old, new in zip(
        original, (t for p in source[study.CHOICES[1]] for t in p.tensors()), strict=True
    ):
        assert torch.equal(old, new)
    assert result[1].source is source[study.CHOICES[1]][1].source
    assert result[1].row_positions is source[study.CHOICES[1]][1].row_positions


def test_refuses_padding_or_silent_underfill():
    source = families()
    with pytest.raises(ValueError, match="not frozen target"):
        study.choose_constructed(source, study.CHOICES[:2], 1)


def test_header_values_not_just_shapes_must_match():
    source = families()
    body = list(source[study.CHOICES[1]])
    body[1] = replace(body[1], cos=body[1].cos + 1)
    source[study.CHOICES[1]] = tuple(body)
    with pytest.raises(ValueError, match="not bitwise identical"):
        study.coalesce_headers(source)


def test_signed_zero_is_not_bitwise_header_equivalence():
    source = families()
    packet = source[study.CHOICES[1]][1]
    source[study.CHOICES[1]] = (
        source[study.CHOICES[1]][0],
        replace(packet, sin=torch.full_like(packet.sin, -0.0)),
    )
    assert torch.equal(packet.sin, source[study.CHOICES[1]][1].sin)
    with pytest.raises(ValueError, match="not bitwise identical"):
        study.coalesce_headers(source)


def test_recorded_policy_payload_is_hash_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "clean_source", lambda *args: "source")
    policy = dict(study.POLICY, seed=99)
    study.write_once(
        tmp_path / "plan.json",
        dict(
            source_sha="source",
            policy=policy,
            policy_sha256=study.digest(study.POLICY),
            bindings={},
        ),
    )
    with pytest.raises(ValueError, match="protocol changed"):
        study.load(tmp_path)


def test_rebinding_preserves_every_reconstructed_component():
    torch.manual_seed(13)
    source = families()
    selected_family = study.coalesce_headers(source)
    witness = study.CHOICES[:2]
    selected = tuple(selected_family[key][i] for i, key in enumerate(witness))
    blocks = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                k_proj=torch.nn.Linear(128, 2, bias=False).to(torch.bfloat16),
                v_proj=torch.nn.Linear(128, 2, bias=False).to(torch.bfloat16),
            )
        )
        for _ in range(2)
    ]
    adapter = SimpleNamespace(_language_layers=lambda: blocks)
    with torch.inference_mode():
        study.verify_header_rebinding(adapter, source, witness, selected)
        changed = list(selected)
        changed[1] = replace(changed[1], cos=changed[1].cos + 1)
        with pytest.raises(RuntimeError, match="rebinding changed"):
            study.verify_header_rebinding(adapter, source, witness, tuple(changed))


@pytest.mark.parametrize("witness", [[], ["unknown"], list(study.CHOICES)])
def test_bad_witness_fails_closed(witness):
    with pytest.raises(ValueError, match="invalid frozen"):
        study.choose_constructed(families(), witness, 100)
