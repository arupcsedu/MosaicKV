"""Random-weight CPU multimodal wiring checks; not pretrained evidence."""

import json
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from test_hf_tiny_models import _llava

from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.evaluation.six_mechanism_gate import build_config
from mosaickv.evaluation.six_mechanism_probe import observe_runtime
from mosaickv.types import Precision


class ImageProcessor:
    chat_template = "synthetic-multimodal-cpu-fixture"

    def apply_chat_template(self, *args, **kwargs):
        return "fixture"

    def __call__(self, *args, **kwargs):
        return dict(
            input_ids=torch.tensor([[1, 4, 63, 63, 63, 63, 5, 6]]),
            attention_mask=torch.ones((1, 8), dtype=torch.long),
            pixel_values=torch.zeros((1, 3, 8, 8)),
        )

    def batch_decode(self, token_ids, **kwargs):
        return [" ".join(str(value) for value in token_ids.reshape(-1).tolist())]


def test_six_mechanism_observer_and_real_redecode_on_tiny_multimodal_model(tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    adapter = _llava(kv_heads=2)
    adapter.processor = ImageProcessor()
    protocol = json.loads(
        (
            Path(__file__).parents[2] / "configs/experiments/six_mechanism_feasibility_v1.json"
        ).read_text()
    )
    config = build_config(protocol, "controlled_repair")
    config = replace(
        config,
        model=replace(config.model, precision=Precision.FP32),
        cache=replace(config.cache, budget_value=1536, retention_ratio=0.75),
        forecasting=replace(config.forecasting, prompt_window=4),
        prototypes=replace(config.prototypes, group_size=2),
        residual=replace(config.residual, require_pinned_memory=False),
    )
    request = EvaluationRequest(
        "cpu-six-mechanisms",
        "fixture",
        "synthetic_only",
        build_multimodal_messages(
            "Describe the fixture.", (MediaItem(MediaKind.IMAGE, Image.new("RGB", (8, 8))),)
        ),
        {},
    )
    with torch.inference_mode(), observe_runtime(adapter, config) as operations:
        output = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path).generate(
            request
        )
    trace = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert trace["tier_mode"] == "three_tier"
    assert operations["prefill"] == operations["forecast"] == operations["planner"] == 1
    assert operations["prototype_members"] > 0
    assert operations["repair"] > 0 and output.metrics.repair_count > 0
    assert all(operations["decode_checks"])
    assert operations["rope_position_checks"] and all(operations["rope_position_checks"])
    assert operations["plan_invariants"] and operations["packed_storage_checks"]
    assert operations["repair_physical_cap_checks"]
    assert operations["promoted_kv_exact_checks"] and all(operations["promoted_kv_exact_checks"])
    assert not operations["benchmark_labels_accessed"]
    assert len(trace["generated_token_ids"]) == 16
