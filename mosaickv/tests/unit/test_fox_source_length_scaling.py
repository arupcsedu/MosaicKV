"""CPU regressions for the bounded synthetic source-scaling study."""

from copy import deepcopy
from types import SimpleNamespace

import evicastkv_source_length_scaling as study
import pytest
import torch

from mosaickv.cache_state import Modality, ModalitySpan


def test_fixture_extends_only_label_free_prompt():
    base = dict(
        task="llava_description",
        sample_id="consumed-id",
        prompt="Describe the image.",
        media=[dict(kind="image", path="/scratch/historical.png", sha256="bound")],
    )
    assert study.fixture(base, 0) == base
    changed = study.fixture(base, 128)
    assert changed["prompt"].startswith(base["prompt"])
    assert changed["prompt"].count(study.FILLER) == 128
    assert {k: v for k, v in changed.items() if k != "prompt"} == {
        k: v for k, v in base.items() if k != "prompt"
    }
    with pytest.raises(ValueError, match="label-free"):
        study.fixture(dict(base, reference="forbidden"), 128)
    with pytest.raises(ValueError, match="registered"):
        study.fixture(base, 129)


def test_geometry_uses_actual_processor_tensors_and_context():
    prepared = SimpleNamespace(
        logical_sequence_length=10,
        modality_map=(
            ModalitySpan(0, 1, Modality.TEXT),
            ModalitySpan(1, 5, Modality.IMAGE, grid_shape=(2, 2)),
            ModalitySpan(5, 10, Modality.TEXT),
        ),
        model_inputs=dict(
            input_ids=torch.zeros(1, 10, dtype=torch.long),
            pixel_values=torch.zeros(1, 3, 2, 2),
        ),
    )
    result = study.geometry(prepared, 200)
    assert result["decoder_prompt_positions"] == 10
    assert result["text_positions"] == 6
    assert result["image_positions"] == 4
    assert result["image_spans"][0]["patch_grid"] == (2, 2)
    with pytest.raises(ValueError, match="exceeds model context"):
        study.geometry(prepared, 136)
    prepared.modality_map = prepared.modality_map[1:]
    with pytest.raises(ValueError, match="contiguous"):
        study.geometry(prepared, 200)


def test_sampled_timing_requires_each_layer_and_measured_shape():
    event = SimpleNamespace(elapsed_time=lambda _: 0.5)
    records = [
        (step, 600, 0.1, event, event) for step in study.FILL_STEPS for _ in range(study.LAYERS)
    ]
    assert study.summarize_fills(records, 600)[0] == {
        "step": 1,
        "layers": 32,
        "host_ms": pytest.approx(3.2),
        "stream_ms": 16.0,
    }
    with pytest.raises(ValueError, match="full prompt fill"):
        study.summarize_fills(records[:-1], 600)
    with pytest.raises(ValueError, match="full prompt fill"):
        study.summarize_fills(records, 601)


def test_prompt_probe_does_not_count_generated_history(monkeypatch):
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.graphed_storage import SharedSourceLayer

    prompt = SimpleNamespace(shape=(1, 32, 600, 128))
    history = SimpleNamespace(shape=(1, 32, 16, 128))
    layer = SimpleNamespace(representation=prompt, history_length=32)
    calls = []

    def fill(packet):
        calls.append(packet)

    def fill_immutable(self):
        ExactFormat.fill(self.representation)
        ExactFormat.fill(history)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True

        def record(self):
            pass

    monkeypatch.setattr(ExactFormat, "fill", fill)
    monkeypatch.setattr(SharedSourceLayer, "fill_immutable", fill_immutable)
    with study.prompt_fills(SimpleNamespace(cuda=SimpleNamespace(Event=Event))) as events:
        SharedSourceLayer.fill_immutable(layer)
        layer.history_length = 33
        SharedSourceLayer.fill_immutable(layer)
    assert calls == [prompt, history, prompt, history]
    assert len(events) == 1
    assert events[0][:2] == (32, 600)
    assert ExactFormat.fill is fill
    assert SharedSourceLayer.fill_immutable is fill_immutable


def records():
    return [
        dict(
            extension_units=units,
            arm=arm,
            repetition=rep,
            warmup=rep < 0,
            fixture_sha256=str(units),
            first_prefill_logit_sha256=str(units),
            geometry=dict(
                decoder_prompt_positions=600 + units,
                image_positions=576,
                processed_pixel_sha256="same",
            ),
            storage={"initial_active_kv_storage_bytes": 100 + units},
            representation_sha256=arm,
            prompt_fill_by_step=[
                dict(step=step, stream_ms=1.0, host_ms=2.0) for step in study.FILL_STEPS
            ],
            status="completed",
        )
        for units in study.EXTENSIONS
        for arm in study.ARMS
        for rep in range(-study.WARMUPS, study.MEASURED)
    ]


def test_aggregate_retains_all_lengths_and_no_best_arm_selection():
    rows = records()
    out = study.summarize(rows)
    assert len(out) == 9
    assert {x["geometry"]["decoder_prompt_positions"] for x in out} == {600, 728, 856}
    assert all(x["measured_repetitions"] == 2 for x in out)
    assert all(x["median_prompt_fill_stream_ms_per_sampled_step"] == 1 for x in out)
    with pytest.raises(ValueError, match="duplicate"):
        study.summarize([*rows[:-1], rows[0]])


@pytest.mark.parametrize("field", ["geometry", "first_prefill_logit_sha256", "fixture_sha256"])
def test_cross_arm_mismatch_blocks(field):
    rows = deepcopy(records())
    rows[0][field] = "drifted"
    with pytest.raises(ValueError, match="drifted"):
        study.summarize(rows)


def test_scaling_requires_unchanged_visual_identity_across_lengths():
    rows = records()
    for row in rows:
        if row["extension_units"] == 256:
            row["geometry"]["processed_pixel_sha256"] = "different"
    with pytest.raises(ValueError, match="processed image stays fixed"):
        study.summarize(rows)


def test_failed_or_incomplete_worker_emits_negative_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "load", lambda _: {"source_sha": "source", "expected_rows": 18})
    gate = study.analyze(tmp_path)
    assert gate["status"] == "blocked"
    assert gate["rows"] == 0
    assert gate["failures"]
    assert gate["headline_latency_eligible"] is False
    assert gate["full_request_KV_only_peak_certified"] is False
    assert gate["fresh_confirmation_authorized"] is False
