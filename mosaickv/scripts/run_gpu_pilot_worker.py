#!/usr/bin/env python3
"""Run every diagnostic pilot cell for one task while reusing one HF model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import torch

from mosaickv.adapters.huggingface import HuggingFaceMultimodalAdapter, load_hf_adapter
from mosaickv.config import config_sha256, load_config
from mosaickv.evaluation.lmms_adapter import run_lmms_development_evaluation
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.evaluation.storage import load_jsonl
from mosaickv.hf_cli import _model_kwargs, run_hf_evaluation


def _decode(processor: Any, token_ids: Any) -> str:
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None:
        decoder = getattr(getattr(processor, "tokenizer", processor), "batch_decode", None)
    if decoder is None:
        raise RuntimeError("processor has no batch_decode implementation")
    decoded = decoder(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answer = str(decoded[0]) if decoded else ""
    if not answer.strip():
        decoded = decoder(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        answer = str(decoded[0]) if decoded else ""
    if not answer.strip():
        raise RuntimeError("ordinary model.generate decoded an empty answer")
    return answer


class OrdinaryGenerateReference:
    """Evaluation-only GenerationMixin reference; never used by MosaicKV."""

    backend = "huggingface"
    method = "ordinary_model_generate_reference"
    retention_ratio = 1.0

    def __init__(
        self,
        adapter: HuggingFaceMultimodalAdapter,
        model_id: str,
        max_new_tokens: int,
    ) -> None:
        self.adapter = adapter
        self.model_id = model_id
        self.supports_video = adapter.capabilities.video
        self.max_new_tokens = max_new_tokens

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        prepared = self.adapter.prepare_inputs(request.messages)
        generation_config = copy.deepcopy(self.adapter.model.generation_config)
        generation_config._from_model_config = False
        generation_config.do_sample = False
        generation_config.max_new_tokens = self.max_new_tokens
        generation_config.min_new_tokens = None
        # Transformers 4.49 interprets ``None`` as "fall back to the model's
        # EOS IDs" when an explicit GenerationConfig is supplied.  An empty
        # list is the audited way to disable EOS stopping for the controlled
        # fixed-length comparison.
        generation_config.eos_token_id = []
        generation_config.forced_eos_token_id = None
        generation_config.forced_bos_token_id = None
        generation_config.suppress_tokens = None
        generation_config.begin_suppress_tokens = None
        generation_config.bad_words_ids = None
        generation_config.sequence_bias = None
        generation_config.repetition_penalty = 1.0
        generation_config.no_repeat_ngram_size = 0
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = 0
        torch.cuda.synchronize(self.adapter.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.adapter.model.generate(
                **dict(prepared.model_inputs),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
        torch.cuda.synchronize(self.adapter.device)
        elapsed = time.perf_counter() - started
        if len(output.scores) != self.max_new_tokens:
            raise RuntimeError(
                "ordinary generate did not honor fixed output length: "
                f"{len(output.scores)} != {self.max_new_tokens}"
            )
        token_ids = output.sequences[:, -len(output.scores) :]
        if int(token_ids.shape[-1]) != self.max_new_tokens:
            raise RuntimeError("ordinary generate silently returned fewer than the fixed length")
        return ModelGeneration(
            _decode(self.adapter.processor, token_ids),
            GenerationMetrics(
                end_to_end_time=elapsed,
                generated_tokens=self.max_new_tokens,
            ),
            effective_method=self.method,
        )


def _load_job(index: Path, array_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads((index.parent / "pilot_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(index.read_bytes()).hexdigest() != manifest.get("index_sha"):
        raise RuntimeError("pilot index SHA does not match its immutable manifest")
    if not 0 <= array_index < len(rows) or rows[array_index].get("array_index") != array_index:
        raise IndexError(f"pilot array index {array_index} is invalid")
    return cast("dict[str, Any]", rows[array_index]), cast("dict[str, Any]", manifest)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _ordinary_reference(
    job: dict[str, Any],
    adapter: HuggingFaceMultimodalAdapter,
    first_config: Any,
) -> dict[str, object]:
    revision = int(job["pilot_revision"])
    run_id = f"ordinary-generate-reference-v{revision}-{job['task_key']}"
    root = Path(job["output_root"]) / "ordinary_reference" / run_id
    root.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "provenance.json"
    raw_path = root / "results.jsonl"
    parquet_path = root / "aggregate.parquet"
    provenance = {
        "classification": "evaluation_only_ordinary_generation_reference",
        "git_sha": _git_sha(),
        "model_id": first_config.model.id,
        "model_revision": first_config.model.revision,
        "dataset_id": first_config.dataset.id,
        "dataset_revision": first_config.dataset.revision,
        "backend": "huggingface",
        "attention_implementation": "eager",
        "precision": first_config.model.precision.value,
        "seed": int(job["seed"]),
        "max_new_tokens": first_config.generation.max_new_tokens,
    }
    if provenance_path.exists() and parquet_path.exists():
        rows = load_jsonl(raw_path)
        if len(rows) != int(job["subset_size"]):
            raise RuntimeError("ordinary reference has an incomplete immutable result set")
        return {
            "run_id": run_id,
            "tasks": [str(job["task"])],
            "selected_samples": len(rows),
            "resumed_samples": len(rows),
            "raw_output": str(raw_path.resolve()),
            "parquet_output": str(parquet_path.resolve()),
            "dataset_revisions": {str(job["dataset_id"]): str(job["dataset_revision"])},
        }
    provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
    summary = run_lmms_development_evaluation(
        run_id=run_id,
        task_names=(str(job["task"]),),
        model=OrdinaryGenerateReference(
            adapter,
            first_config.model.id,
            max_new_tokens=first_config.generation.max_new_tokens,
        ),
        raw_output=raw_path,
        manifest_path=str(provenance_path.resolve()),
        seed=int(job["seed"]),
        subset_size=int(job["subset_size"]),
        parquet_output=parquet_path,
        dataset_revision=str(job["dataset_revision"]),
    )
    return cast("dict[str, object]", summary)


def run(index: Path, array_index: int) -> int:
    job, manifest = _load_job(index, array_index)
    cells = cast("list[dict[str, Any]]", job["cells"])
    configs = [load_config(cell["config_path"]) for cell in cells]
    for cell, config in zip(cells, configs, strict=True):
        if config_sha256(config) != cell["config_sha"]:
            raise RuntimeError(f"config SHA mismatch: {cell['config_path']}")
        if config.dataset.revision != job["dataset_revision"]:
            raise RuntimeError("worker config dataset revision mismatch")
    first = configs[0]
    adapter = load_hf_adapter(
        first.model.id,
        revision=first.model.revision,
        model_kwargs=_model_kwargs(first, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    output_root = Path(job["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    try:
        summaries.append(_ordinary_reference(job, adapter, first))
    except Exception as error:  # preserve task-level reference failure and continue cells
        errors.append(
            {
                "run_id": (
                    f"ordinary-generate-reference-v{int(job['pilot_revision'])}-{job['task_key']}"
                ),
                "error": f"{type(error).__name__}: {error}",
            }
        )
    for cell, config in zip(cells, configs, strict=True):
        args = argparse.Namespace(
            task=job["task"],
            run_id=cell["run_id"],
            output_dir=job["output_root"],
            raw_output=None,
            parquet_output=None,
            manifest=None,
            trace_directory=job["trace_directory"],
            subset_size=job["subset_size"],
            local_files_only=True,
        )
        try:
            status, summary = run_hf_evaluation(args, config, adapter=adapter)
            summary["requested_method"] = str(cell["requested_method"])
            summary["retention_ratio"] = float(cell["retention_ratio"])
            summary["role"] = str(cell["role"])
            summaries.append(cast("dict[str, object]", summary))
            if status:
                errors.append(
                    {
                        "run_id": str(cell["run_id"]),
                        "error": f"evaluation returned status {status}",
                    }
                )
        except Exception as error:  # keep the remaining pilot cells runnable
            errors.append(
                {
                    "run_id": str(cell["run_id"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    worker_record = {
        "array_index": array_index,
        "classification": manifest["classification"],
        "errors": errors,
        "job_id": job.get("task_key"),
        "summaries": summaries,
    }
    worker_path = output_root / "workers" / f"{array_index:02d}-{job['task_key']}.json"
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_path.write_text(json.dumps(worker_record, sort_keys=True) + "\n", encoding="utf-8")
    return int(bool(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--array-index", required=True, type=int)
    args = parser.parse_args()
    status = run(args.index.resolve(), args.array_index)
    if not math.isfinite(float(status)):
        raise RuntimeError("invalid worker status")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
