import pytest
import torch

from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.shared_head_scope import choose_group


def test_source_groups_obey_text_quantizer_and_all_head_demand():
    torch.manual_seed(32)
    original = torch.zeros(1, 1, 256, 64).bfloat16()
    original[:, :, 8:16] = torch.randn(1, 1, 8, 64).bfloat16()
    body = pack(original, bits=3, axis=2)
    group, proof = choose_group(body, tuple(range(16, 256)), torch.ones(256), 8)
    assert group == tuple(range(8))  # zero source distortion, source-index tie
    assert proof["risk"] == 0
    assert proof["not_the_six_component_submodular_planner"]
    with pytest.raises(ValueError):
        choose_group(body, tuple(range(256)), torch.ones(256), 8)


def test_declared_source_risk_changes_with_prompt_demand_without_label_access():
    torch.manual_seed(42)
    body = pack(torch.randn(1, 1, 128, 64).bfloat16(), bits=3, axis=2)
    demand = torch.ones(128)
    demand[8:16] = 0
    group, _ = choose_group(body, (), demand, 8)
    assert group == tuple(range(8, 16))


def test_shared_map_preserves_interlayer_alignment_and_holds_no_old_codes(monkeypatch):
    import gc
    from dataclasses import dataclass

    from mosaickv.evaluation import shared_source_format as formats
    from mosaickv.evaluation import streaming_source_constructor as constructor
    from mosaickv.evaluation.shared_head_scope import use

    @dataclass
    class Packet:
        source: object
        shape: tuple

    def base_construct(normalized, *_args, **_kwargs):
        return Packet(pack(normalized, bits=3, axis=2), (1, 2, 128, 32))

    def call(self, index, attention, original, args, kwargs, output):
        return formats.construct(kwargs["normalized"], original, (), (127,), None, None)

    monkeypatch.setattr(formats, "construct", base_construct)
    monkeypatch.setattr(constructor.StreamingDirectSource, "__call__", call)
    torch.manual_seed(5)
    with use(group_size=8, map_policy="first_layer_shared") as report:
        encoder = object.__new__(constructor.StreamingDirectSource)
        packets = []
        for i in range(32):
            packet = encoder(
                i,
                None,
                None,
                (),
                {"normalized": torch.randn(1, 1, 128, 64).bfloat16()},
                (None, torch.ones(1, 2, 16, 128)),
            )
            packets.append(packet)
        assert len({p.source.row_map.data_ptr() for p in packets}) == 1
        assert all(p.source.row_map_shared for p in packets)
    gc.collect()
    assert report["all_original_source_tensor_objects_retired"]
    assert len({tuple(row["prototype_groups"][0]) for row in report["layers"]}) == 1
