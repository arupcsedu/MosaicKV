"""Tiny random VLM; no downloads or GPU, not a quality measurement."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from PIL import Image
from test_hf_tiny_models import _baseline_runtime_config, _llava

from mosaickv.config import DatasetConfig, GenerationConfig
from mosaickv.evaluation.paper_native_lanes import load_paper_native_ledger
from mosaickv.evaluation.tasks import default_task_registry
from mosaickv.evaluation.worker_support import record_hf_run
from mosaickv.types import MosaicKVMethod


@pytest.mark.parametrize("reference_count", [5, 6, 7])
def test_materialized_coco_cli_metrics_references_and_storage(tmp_path, reference_count):
    torch.manual_seed(19)
    adapter = _llava()
    spec = default_task_registry().resolve("coco_caption")
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    sample = dict(
        sample_id="one",
        prompt="Describe this image.",
        references=[f"a cat, annotation {index}" for index in range(reference_count)],
        media=[
            dict(kind="image", path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        ],
    )
    payload = dict(
        schema_version=1,
        task=spec.name,
        dataset_id=spec.dataset_id,
        dataset_revision="b" * 40,
        split=spec.split,
        samples=[sample],
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(payload))
    sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    cfg = replace(
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        dataset=DatasetConfig(spec.dataset_id, "b" * 40, spec.split),
        generation=GenerationConfig(max_new_tokens=4),
    )
    result = record_hf_run(
        cfg,
        adapter,
        task=spec.name,
        run_id="coco",
        output_root=tmp_path / "out",
        ledger_path=ledger,
        ledger_sha=sha,
        sample_ids=("one",),
        phase="synthetic_fixture",
        repetition=0,
        order=0,
        export_metrics=True,
        materialized_coco=True,
    )
    assert result["rows"][0]["task_score"] is None
    assert json.loads(result["rows"][0]["reference"]) == sample["references"]
    assert result["rows"][0]["initial_active_kv_storage_bytes"] > 0
    metric = json.loads((tmp_path / "out/coco/metrics.jsonl").read_text())
    assert metric["metric_name"] == "cider" and metric["metric_scope"] == "corpus"
    assert metric["reference_artifact_sha"] == sha
    manifest = json.loads((tmp_path / "out/coco/manifest.json").read_text())
    assert manifest["result_category"] == "unified_comparison"
    trace = json.loads(next((tmp_path / "out/traces/coco").glob("*.json")).read_text())
    assert len(trace["source_kv_layout"]) > 0
    assert (
        trace["cache_lifecycle"]["final_active_kv_storage_bytes"]
        >= result["rows"][0]["initial_active_kv_storage_bytes"]
    )
    payload["samples"][0]["references"] = ["not all annotations"]
    ledger.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="five"):
        load_paper_native_ledger(
            ledger,
            expected_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
            task=spec,
            dataset_revision="b" * 40,
        )
