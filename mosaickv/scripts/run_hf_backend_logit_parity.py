#!/usr/bin/env python3
"""Measure same-prefix FullKV logit deviation across HF attention backends."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import torch

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import RunConfig, config_sha256, load_config
from mosaickv.evaluation.lmms_adapter import run_lmms_development_evaluation
from mosaickv.evaluation.model import EvaluationRequest, ModelGeneration
from mosaickv.evaluation.tasks import default_task_registry
from mosaickv.hf_cli import _model_kwargs
from mosaickv.manifest import (
    ArtifactProvenance,
    InputProvenance,
    RunManifestWriter,
    sha256_bytes,
)
from mosaickv.types import MeasurementType


class _RequestCaptureModel:
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    backend = "request_capture"
    method = "full_kv"
    retention_ratio = 1.0
    supports_video = True

    def __init__(self) -> None:
        self.requests: list[EvaluationRequest] = []

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        self.requests.append(request)
        return ModelGeneration("B", effective_method="request_capture_only")


def _backend_cells(index: Path) -> dict[str, dict[str, Any]]:
    jobs = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    cells = {
        str(cell["cell_id"]): cast("dict[str, Any]", cell)
        for job in jobs
        if job.get("job_kind") == "backend_parity"
        for cell in cast("list[dict[str, Any]]", job["cells"])
    }
    required = {"full_eager", "full_sdpa", "full_fa2"}
    if not required.issubset(cells):
        raise RuntimeError(
            f"evaluation index lacks FullKV backend cells: {sorted(required - cells)}"
        )
    return cells


def _capture_request(config: RunConfig, sample_id: str) -> EvaluationRequest:
    capture = _RequestCaptureModel()
    with tempfile.TemporaryDirectory(prefix="exp5-logit-request-") as temporary:
        root = Path(temporary)
        run_lmms_development_evaluation(
            run_id="exp5-logit-request-capture",
            task_names=("mmstar",),
            model=capture,
            raw_output=root / "capture.jsonl",
            manifest_path=str(root / "capture-manifest.json"),
            seed=config.execution.seed,
            subset_size=None,
            explicit_sample_ids=(sample_id,),
            parquet_output=root / "capture.parquet",
            registry=default_task_registry(),
            dataset_revision=config.dataset.revision,
        )
    if len(capture.requests) != 1 or capture.requests[0].sample_id != sample_id:
        raise RuntimeError("lmms request capture did not return the pinned sample")
    return capture.requests[0]


def _last_logits(logits: Any) -> torch.Tensor:
    return logits.detach().float().reshape(-1, int(logits.shape[-1]))[-1].cpu()


def _trajectory(
    adapter: Any,
    request: EvaluationRequest,
    attention: str,
    max_new_tokens: int,
    *,
    forced_tokens: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[torch.Tensor, ...]]:
    adapter.set_attention_implementation(attention)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    prepared = adapter.prepare_inputs(request.messages)
    prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    predicted = [int(prefill.next_token_id.detach().cpu().reshape(-1)[0].item())]
    logits = [_last_logits(prefill.logits)]
    state = prefill.state
    token = prefill.next_token_id
    for step in range(1, max_new_tokens):
        if forced_tokens is not None:
            token = torch.tensor(
                [[forced_tokens[step - 1]]],
                dtype=torch.long,
                device=adapter.device,
            )
        output = adapter.decode_one_token(
            token,
            state,
            capture_queries=False,
            capture_attentions=False,
        )
        state = output.state
        token = output.next_token_id
        predicted.append(int(token.detach().cpu().reshape(-1)[0].item()))
        logits.append(_last_logits(output.logits))
    return tuple(predicted), tuple(logits)


def _comparison(
    reference_tokens: tuple[int, ...],
    reference_logits: tuple[torch.Tensor, ...],
    candidate_tokens: tuple[int, ...],
    candidate_logits: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    if len(reference_logits) != len(candidate_logits):
        raise RuntimeError("backend trajectories have different lengths")
    per_step: list[dict[str, Any]] = []
    absolute_sum = 0.0
    element_count = 0
    for step, (reference, candidate) in enumerate(
        zip(reference_logits, candidate_logits, strict=True)
    ):
        if reference.shape != candidate.shape:
            raise RuntimeError(f"backend logit shape mismatch at step {step}")
        difference = torch.abs(reference - candidate)
        kl = torch.sum(
            torch.softmax(reference, dim=-1)
            * (torch.log_softmax(reference, dim=-1) - torch.log_softmax(candidate, dim=-1))
        )
        absolute_sum += float(difference.sum().item())
        element_count += int(difference.numel())
        per_step.append(
            {
                "step": step,
                "maximum_absolute_logit_difference": float(difference.max().item()),
                "mean_absolute_logit_difference": float(difference.mean().item()),
                "reference_to_candidate_logit_kl": float(kl.item()),
                "reference_token_id": reference_tokens[step],
                "candidate_same_prefix_token_id": candidate_tokens[step],
            }
        )
    agreements = [
        left == right for left, right in zip(reference_tokens, candidate_tokens, strict=True)
    ]
    return {
        "comparison_semantics": "full_vocabulary_logits_on_identical_eager_reference_prefixes",
        "generated_steps": len(reference_tokens),
        "vocabulary_size": int(reference_logits[0].numel()),
        "maximum_absolute_logit_difference": max(
            float(step["maximum_absolute_logit_difference"]) for step in per_step
        ),
        "mean_absolute_logit_difference": absolute_sum / element_count,
        "mean_reference_to_candidate_logit_kl": sum(
            float(step["reference_to_candidate_logit_kl"]) for step in per_step
        )
        / len(per_step),
        "same_prefix_token_agreement": sum(agreements) / len(agreements),
        "reference_token_ids": list(reference_tokens),
        "candidate_same_prefix_token_ids": list(candidate_tokens),
        "per_step": per_step,
    }


def _write_result(
    destination: Path,
    payload: dict[str, Any],
    config: RunConfig,
    inputs: InputProvenance,
) -> tuple[Path, Path]:
    run_id = str(payload["run_id"])
    run_root = destination / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    result_path = run_root / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256_bytes(result_path.read_bytes())
    manifest_path = RunManifestWriter().write(
        run_root / "manifest.json",
        config,
        MeasurementType.VALIDATION_SMOKE,
        inputs,
        ArtifactProvenance(digest, "not_applicable", digest),
        run_id=run_id,
        execution_metadata={
            "comparison_semantics": payload["comparison_semantics"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_used"),
        },
    )
    return result_path, manifest_path


def run(index: Path, destination: Path, sample_id: str) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite backend logit probe: {destination}")
    cells = _backend_cells(index)
    configs = {name: load_config(cells[name]["config_path"]) for name in cells}
    for name, config in configs.items():
        if config_sha256(config) != cells[name]["config_sha"]:
            raise RuntimeError(f"backend logit config SHA mismatch: {name}")
    eager = configs["full_eager"]
    request = _capture_request(eager, sample_id)
    adapter = load_hf_adapter(
        eager.model.id,
        revision=eager.model.revision,
        model_kwargs=_model_kwargs(eager, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    reference_tokens, reference_logits = _trajectory(
        adapter,
        request,
        "eager",
        eager.generation.max_new_tokens,
        forced_tokens=None,
    )
    source_manifest_path = (
        Path("/scratch/djy8hg/runs/mosaickv/generalization-evaluation-v1")
        / cells["full_eager"]["run_id"]
        / "manifest.json"
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    inputs = InputProvenance(**source_manifest["inputs"])
    destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for cell_id, attention in (
        ("full_eager", "eager"),
        ("full_sdpa", "sdpa"),
        ("full_fa2", "flash_attention_2"),
    ):
        if attention == "eager":
            candidate_tokens, candidate_logits = reference_tokens, reference_logits
        else:
            candidate_tokens, candidate_logits = _trajectory(
                adapter,
                request,
                attention,
                eager.generation.max_new_tokens,
                forced_tokens=reference_tokens,
            )
        record = {
            "schema_version": 1,
            "run_id": f"exp5-backend-logit-{attention}",
            "sample_id": sample_id,
            "model": eager.model.id,
            "model_revision": eager.model.revision,
            "precision": eager.model.precision.value,
            "reference_attention": "eager",
            "candidate_attention": attention,
            "reference_config_sha": cells["full_eager"]["config_sha"],
            "candidate_config_sha": cells[cell_id]["config_sha"],
            **_comparison(
                reference_tokens,
                reference_logits,
                candidate_tokens,
                candidate_logits,
            ),
        }
        result_path, manifest_path = _write_result(
            destination,
            record,
            configs[cell_id],
            inputs,
        )
        records.append(
            {
                **record,
                "result_path": str(result_path.resolve()),
                "result_sha": sha256_bytes(result_path.read_bytes()),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha": sha256_bytes(manifest_path.read_bytes()),
            }
        )
    summary = {
        "schema_version": 1,
        "status": "passed",
        "measurement": "same_prefix_full_vocabulary_fullkv_backend_logit_parity",
        "records": records,
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--sample-id", default="mmstar:1343")
    args = parser.parse_args()
    result = run(args.index.resolve(), args.destination.resolve(), args.sample_id)
    print(json.dumps({"records": len(result["records"]), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
