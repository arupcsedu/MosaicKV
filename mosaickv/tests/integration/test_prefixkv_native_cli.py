"""End-to-end synthetic CPU native route; no checkpoint/model download."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image  # noqa: E402
from test_hf_tiny_models import _baseline_runtime_config, _llava  # noqa: E402

from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel  # noqa: E402
from mosaickv.config import DatasetConfig, GenerationConfig  # noqa: E402
from mosaickv.evaluation.tasks import default_task_registry  # noqa: E402
from mosaickv.evaluation.worker_support import record_hf_run  # noqa: E402
from mosaickv.types import MosaicKVMethod  # noqa: E402


def test_teacher_forcing_rejects_over_limit_instead_of_truncating(tmp_path):
    adapter = _llava()
    config = _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0)
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path)
    # The synthetic tokenizer has four target IDs; the limit is two.
    with pytest.raises(ValueError, match="rather than truncating"):
        runtime._teacher_forced_target_ids("human target")


def test_teacher_forcing_scores_only_target_tokens_not_prompt(tmp_path):
    from mosaickv.evaluation.model import EvaluationRequest

    torch.manual_seed(71)
    adapter = _llava()
    config = replace(
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        generation=GenerationConfig(max_new_tokens=4),
    )
    request = EvaluationRequest(
        "mask", "sample", "llava_description", [{"role": "user", "content": "longer prompt"}], {}
    )
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=tmp_path)
    targets = runtime._teacher_forced_target_ids("target")
    prepared = adapter.prepare_inputs(request.messages)
    prefill = adapter.prefill(prepared)
    logits = prefill.logits
    state = prefill.state
    expected = 0.0
    for index, target in enumerate(targets):
        token = torch.tensor([[target]], device=logits.device)
        expected += float(torch.nn.functional.cross_entropy(logits.float(), token[0]).item())
        if index < len(targets) - 1:
            decoded = adapter.decode_one_token(token, state)
            logits, state = decoded.logits, decoded.state
    measured = runtime.teacher_forced_nll(request, "target")
    assert measured.evaluated_tokens == len(targets) == 4
    assert measured.negative_log_likelihood_sum == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("task", ["llava_description", "mmvet_prefixkv"])
def test_unified_cli_full_reference_then_topk_metrics_and_manifest(tmp_path, task):
    torch.manual_seed(19)
    adapter = _llava()
    spec = default_task_registry().resolve(task)
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(image)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            dict(
                schema_version=1,
                task=task,
                dataset_id=spec.dataset_id,
                dataset_revision="b" * 40,
                split=spec.split,
                samples=[
                    dict(
                        sample_id="one",
                        prompt="Describe this image.",
                        references=["target"],
                        teacher_forcing_target="human target",
                        media=[
                            dict(
                                kind="image",
                                path=str(image),
                                sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                            )
                        ],
                    )
                ],
            )
        )
    )
    config = replace(
        _baseline_runtime_config(MosaicKVMethod.FULL_KV, 1.0),
        dataset=DatasetConfig(spec.dataset_id, "b" * 40, spec.split),
        generation=GenerationConfig(max_new_tokens=4),
    )
    reference = tmp_path / "reference.json"
    common = dict(
        adapter=adapter,
        task=task,
        output_root=tmp_path / "outputs",
        ledger_path=ledger,
        ledger_sha=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        sample_ids=("one",),
        phase="synthetic_ci_not_measured_evidence",
        repetition=0,
        order=0,
        export_metrics=True,
        materialized_prefixkv=True,
    )
    full = record_hf_run(config, **common, run_id="full", fullkv_reference_output=reference)
    prompt = replace(
        config,
        method=MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        cache=replace(config.cache, retention_ratio=0.5),
    )
    candidate = record_hf_run(
        prompt,
        **common,
        run_id="topk",
        fullkv_reference_input=reference,
        fullkv_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
    )
    assert full["rows"][0]["status"] == candidate["rows"][0]["status"] == "completed"
    for name in ("full", "topk"):
        manifest = json.loads((tmp_path / "outputs" / name / "manifest.json").read_text())
        assert manifest["result_category"] == "unified_comparison"
        metrics = [
            json.loads(line)
            for line in (tmp_path / "outputs" / name / "metrics.jsonl").read_text().splitlines()
        ]
        assert {m["metric_name"] for m in metrics} == {"rouge_l_f1", "teacher_forced_ppl"}
        assert all(m["result_category"] == "unified_comparison" for m in metrics)
        sample_ppl = next(
            m
            for m in metrics
            if m["metric_name"] == "teacher_forced_ppl" and m["metric_scope"] == "sample"
        )
        assert sample_ppl["evaluated_token_count"] == 4
        traces = list((tmp_path / "outputs/traces" / name).glob("*.json"))
        # Generation and teacher forcing must preserve separate traces.
        assert len(traces) == 2
