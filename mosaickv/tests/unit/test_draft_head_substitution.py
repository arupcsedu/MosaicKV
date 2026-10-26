"""Synthetic draft-evidence correctness, not checkpoint-quality validation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_head_substitution import fixture

from mosaickv.backends.hf_draft_head_evidence import DraftAttentionRecorder
from mosaickv.cache_state import FullKVState
from mosaickv.selection.draft_head_substitution import DraftHeadPoolPolicy, draft_substitution_pool
from mosaickv.selection.head_substitution import pack_substitution


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def synthetic():
    original, anchor, _ = fixture()
    rng = torch.Generator().manual_seed(17)
    values = [torch.randn(layer.value.shape, generator=rng) for layer in original.layers]
    full = FullKVState.from_tensors(
        tuple((layer.key, values[i]) for i, layer in enumerate(original.layers)),
        block_size=1,
        logical_positions=(10, 20, 30, 40, 50, 60),
        mandatory_logical_positions=(60,),
        original_logical_sequence_length=61,
        next_decode_position=61,
    )
    anchor = replace(
        anchor,
        snapshot=replace(
            anchor.snapshot,
            layers=tuple(
                replace(layer, value=values[i][:, :, [0, 2, 5]].clone())
                for i, layer in enumerate(anchor.snapshot.layers)
            ),
        ),
    )
    evidence = []
    for layer in full.layers:
        p = torch.softmax(torch.randn((2, 4, 6), generator=rng), -1) * 0.8
        tail = torch.randn((2, 4, 4), generator=rng) * 0.2
        evidence.append(
            dict(
                prompt_probabilities=p,
                target=torch.bmm(p, layer.value[0]) + tail,
                generated_numerator=tail,
                generated_mass=torch.full((2, 4), 0.2),
            )
        )
    return full, anchor, tuple(evidence)


def loss(data, values, positions, head):
    p = data["prompt_probabilities"][head, :, positions]
    numerator = p @ values[head, positions] + data["generated_numerator"][head]
    denominator = p.sum(-1) + data["generated_mass"][head]
    return float((numerator / denominator[:, None] - data["target"][head]).square().mean())


def test_draft_pairs_match_brute_force_preserve_exact_bytes_and_mandatory():
    full, anchor, evidence = synthetic()
    pool, trace = draft_substitution_pool(full, anchor, evidence, limit=16)
    assert pool and (pool, trace) == draft_substitution_pool(full, anchor, evidence, limit=16)
    assert (pool, trace) == draft_substitution_pool(
        full, anchor, evidence, limit=16, policy=DraftHeadPoolPolicy(position_chunk_size=2)
    )
    expected = []
    for i, source in enumerate(full.layers):
        for h in range(2):
            before = loss(evidence[i], source.value[0], [0, 2, 5], h)
            pairs = []
            for d in (0, 2):
                for a in (1, 3, 4):
                    positions = sorted(({0, 2, 5} - {d}) | {a})
                    pairs.append((loss(evidence[i], source.value[0], positions, h), d, a))
            after, d, a = min(pairs)
            if after < before:
                expected.append((before - after, i, h, d, a))
    expected.sort(key=lambda row: (-row[0], *row[1:]))
    assert [(p.layer, p.head, p.removed_position, p.added_position) for p in pool] == [
        (i, h, d, a) for _, i, h, d, a in expected
    ]
    for p, t, (_, i, h, d, a) in zip(pool, trace, expected, strict=True):
        assert t["draft_output_error_after"] == pytest.approx(
            loss(evidence[i], full.layers[i].value[0], sorted(({0, 2, 5} - {d}) | {a}), h), abs=1e-6
        )
        assert t["predicted_gain"] > 0
        changed = pack_substitution(full, anchor, p)
        assert [layer.key.shape for layer in changed.snapshot.layers] == [
            layer.key.shape for layer in anchor.snapshot.layers
        ]
        assert p.removed_position != 5
        assert p.replaced_bytes == 32
        for j, layer in enumerate(changed.snapshot.layers):
            for head in range(2):
                if (j, head) != (p.layer, p.head):
                    assert torch.equal(
                        layer.value[:, head], anchor.snapshot.layers[j].value[:, head]
                    )


def test_individual_draft_queries_are_not_averaged_before_loss_and_no_prompt_argument():
    full, anchor, evidence = synthetic()
    _, trace = draft_substitution_pool(full, anchor, evidence, limit=16)
    for t in trace:
        i, h = t["layer"], t["kv_head"]
        assert t["draft_output_error_before"] == pytest.approx(
            loss(evidence[i], full.layers[i].value[0], [0, 2, 5], h), abs=1e-6
        )
    pooled = tuple({key: t.mean(1, keepdim=True) for key, t in row.items()} for row in evidence)
    _, averaged = draft_substitution_pool(full, anchor, pooled, limit=16)
    assert {(t["layer"], t["kv_head"]): t["draft_output_error_before"] for t in trace} != {
        (t["layer"], t["kv_head"]): t["draft_output_error_before"] for t in averaged
    }
    with pytest.raises(TypeError):
        draft_substitution_pool(full, anchor, evidence, limit=16, benchmark_answers=["forbidden"])


@pytest.mark.parametrize("bad", ["nan", "shape", "mandatory", "duplicate"])
def test_invalid_evidence_and_membership_fail_closed(bad):
    full, anchor, evidence = synthetic()
    if bad == "nan":
        evidence[0]["prompt_probabilities"][0, 0, 0] = float("nan")
    elif bad == "shape":
        evidence[0]["target"] = torch.zeros(1)
    elif bad == "mandatory":
        anchor = replace(
            anchor,
            slot_records=tuple(
                dict(row, logical_position=50) if row["logical_position"] == 60 else row
                for row in anchor.slot_records
            ),
        )
    else:
        anchor = replace(anchor, slot_records=(*anchor.slot_records, anchor.slot_records[0]))
    with pytest.raises(ValueError):
        draft_substitution_pool(full, anchor, evidence, limit=16)


def test_no_positive_surrogate_gain_does_not_create_a_fake_proposal():
    full, anchor, _ = synthetic()
    evidence = tuple(
        dict(
            prompt_probabilities=torch.ones(2, 2, 6) / 7,
            target=torch.zeros(2, 2, 4),
            generated_numerator=torch.zeros(2, 2, 4),
            generated_mass=torch.ones(2, 2) / 7,
        )
        for _ in full.layers
    )
    # Zero original values and target make every exact swap a genuine tie.
    for source in full.layers:
        source.value.zero_()
    pool, trace = draft_substitution_pool(full, anchor, evidence, limit=16)
    assert pool == () and trace == []
    for count in (0, 17):
        with pytest.raises(ValueError):
            draft_substitution_pool(full, anchor, evidence, limit=count)
    with pytest.raises(ValueError):
        DraftHeadPoolPolicy(choices_per_side=9)


def test_recorder_preserves_grouped_query_heads_and_actual_ring_history():
    full, _, _ = synthetic()
    rng = torch.Generator().manual_seed(5)
    extended = tuple(
        (layer.key, torch.cat((layer.value, torch.randn((1, 2, 2, 4), generator=rng)), 2))
        for layer in full.layers
    )
    adapter = SimpleNamespace(_legacy_layers=lambda state: (state, None))
    recorder = DraftAttentionRecorder(adapter, full, horizon=2)
    original = [layer.value.clone() for layer in full.layers]
    for step in (1, 2):
        layers = tuple((k, v[:, :, : 6 + step]) for k, v in extended)
        probabilities = tuple(
            torch.softmax(torch.randn((1, 4, 1, 6 + step), generator=rng), -1) for _ in full.layers
        )
        out = SimpleNamespace(
            state=SimpleNamespace(past_key_values=layers), attention_weights=probabilities
        )
        # Second prefix retains only the latest generated token, not token 61.
        state = SimpleNamespace(
            past_key_values=SimpleNamespace(
                layers=tuple(
                    SimpleNamespace(generated_logical_positions=(60 + step,)) for _ in full.layers
                )
            )
        )
        recorder.observe(step, out, state)
    result = recorder.result()
    assert result[0]["prompt_probabilities"].shape == (2, 4, 6)
    assert recorder.steps[1][0]["retained_generated_physical_indices"] == (7,)
    for i, data in enumerate(result):
        for step in (1, 2):
            p = recorder.steps[step - 1][i]["prompt_probabilities"]
            torch.testing.assert_close(
                data["prompt_probabilities"][:, (step - 1) * 2 : step * 2], p
            )
    assert all(
        torch.equal(source.value, v) for source, v in zip(full.layers, original, strict=True)
    )
    assert recorder.storage_bytes > 0
    with pytest.raises(ValueError, match="ordered"):
        recorder.observe(2, out, state)
    with pytest.raises(ValueError, match="complete"):
        DraftAttentionRecorder(adapter, full, horizon=2).result()


def test_protocol_and_negative_seal_keep_larger_execution_blocked():
    import json
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[3]
    protocol = yaml.safe_load(
        (root / "mosaickv/configs/experiments/bounded_draft_head_v8.yaml").read_text()
    )
    seal = json.loads((root / protocol["negative_seal"]).read_text())
    assert seal["accepted_proposals"] == 0 and seal["v7_1_survivor_rejected_prefixes"] == [5, 5, 6]
    assert protocol["diagnostic"]["guard"]["horizon"] == 32
    assert protocol["diagnostic"]["guard"]["absolute_tolerance"] == 1e-7
    assert protocol["diagnostic"]["guard"]["relative_tolerance"] == 1e-6
    assert protocol["workers"] == 2 and protocol["max_global_a100s"] == 6
    assert all(
        not protocol[k] and not seal[k]
        for k in ("fresh_cohort_authorized", "calibration_authorized", "paper_campaign_authorized")
    )
