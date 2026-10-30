"""EOS-aware model-reference quality diagnostics; not independent benchmark evidence."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.metrics.scorers import rouge_l_f1


def answer_prefix(
    ids: Sequence[int], *, eos: tuple[int, ...], cap: int, decode: Callable
) -> dict[str, Any]:
    if type(cap) is not int or cap < 1 or not eos or any(type(t) is not int for t in (*ids, *eos)):
        raise ValueError("positive cap, explicit EOS IDs and integer prediction IDs required")
    chosen = list(ids[:cap])
    end = next((i for i, token in enumerate(chosen) if token in eos), None)
    tokens = chosen[:end] if end is not None else chosen
    return dict(
        answer=decode(tokens),
        answer_tokens=len(tokens),
        stopped_by_eos=end is not None,
        output_limit_reached=end is None and len(chosen) == cap,
        prefix_token_cap=cap,
        token_ids=tokens,
    )


def load_processor(model: dict[str, Any]) -> tuple[Any, Any, str]:
    from transformers import AutoProcessor, GenerationConfig

    from mosaickv.adapters.huggingface.registry import _resolve_model_source

    snapshot = _resolve_model_source(model["id"], model["revision"], local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    generation = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
    return processor, generation, snapshot


def derive(
    root: Path, expected: str, output: Path, *, source: Path, reference_root: Path | None = None
) -> None:
    import json

    import pandas as pd  # type: ignore[import-untyped]
    import torch

    if output.exists() or sha(root / "plan.json") != expected:
        raise ValueError("new derived root and immutable parent plan required")
    plan = json.loads((root / "plan.json").read_text())
    jobs = plan["jobs"]
    model = jobs[0]["arms"][0]["config"]["model"]
    processor, generation, snapshot = load_processor(model)
    eos = generation.eos_token_id
    eos = (eos,) if type(eos) is int else tuple(eos)

    def decode(ids):
        return processor.tokenizer.decode(ids, skip_special_tokens=True)

    inputs = {str(root / "plan.json"): expected}
    for name in (
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "generation_config.json",
    ):
        path = Path(snapshot) / name
        if not path.is_file():
            raise ValueError("complete pinned processor/reference-scoring files required")
        inputs[str(path)] = sha(path)
    references = {}
    controls = {}
    if reference_root is not None:
        reference_plan_path = reference_root / "plan.json"
        digest = sha(reference_plan_path)
        if plan["bound_files"].get(str(reference_plan_path)) != digest:
            raise ValueError("external FullKV reference plan must be hash-bound")
        inputs[str(reference_plan_path)] = digest
        external = json.loads(reference_plan_path.read_text())
        for ref_job in external["jobs"]:
            if ref_job["method"] != "full_kv":
                continue
            compatible = [
                j
                for j in jobs
                if (j["task"], j["sample_id"]) == (ref_job["task"], ref_job["sample_id"])
            ]
            if not compatible:
                continue
            ref_cfg = ref_job["arms"][0]["config"]
            if any(
                any(
                    j["arms"][0]["config"][k] != ref_cfg[k]
                    for k in ("model", "generation", "execution", "dataset")
                )
                or sha(Path(j["input"])) != sha(Path(ref_job["input"]))
                for j in compatible
            ):
                raise ValueError("external reference controls or input content differ")
            folder = reference_root / f"workers/{ref_job['index']}"
            result_path = folder / "result.json"
            result = json.loads(result_path.read_text())
            if plan["bound_files"].get(str(result_path)) != sha(result_path):
                raise ValueError("external FullKV result must be frozen before execution")
            if (
                result["status"] != "completed"
                or result["failures"]
                or result["source_sha"] != external["source_sha"]
                or result["plan_sha256"] != digest
            ):
                raise ValueError("invalid external FullKV producer")
            inputs[str(result_path)] = sha(result_path)
            for name in ("logits.pt", "manifest.json", "ordinary_parity.json"):
                path = folder / name
                if result["outputs"].get(name) != sha(path):
                    raise ValueError("external FullKV evidence changed")
                if plan["bound_files"].get(str(path)) != sha(path):
                    raise ValueError("external FullKV outputs must be frozen before execution")
                inputs[str(path)] = sha(path)
            logits = torch.load(folder / "logits.pt", map_location="cpu", weights_only=True)[
                "reference"
            ]
            references[(ref_job["task"], ref_job["sample_id"])] = (
                logits.argmax(-1).tolist(),
                sha(folder / "logits.pt"),
            )
    for job in jobs:
        folder = root / f"workers/{job['index']}"
        path = folder / "result.json"
        result = json.loads(path.read_text())
        if (
            result["status"] != "completed"
            or result["failures"]
            or result["source_sha"] != plan["source_sha"]
            or result["plan_sha256"] != expected
        ):
            raise ValueError("complete, producer-bound worker records required")
        inputs[str(path)] = sha(path)
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("worker evidence changed")
            inputs[str(folder / name)] = digest
        if job["method"] == "full_kv":
            logits = torch.load(folder / "logits.pt", map_location="cpu", weights_only=True)[
                "reference"
            ]
            key = (job["task"], job["sample_id"])
            if key in references:
                raise ValueError("duplicate FullKV reference")
            references[key] = (logits.argmax(-1).tolist(), sha(folder / "logits.pt"))
        controls[job["index"]] = json.loads((folder / "precision_controls.json").read_text())
    rows = []
    for job in jobs:
        folder = root / f"workers/{job['index']}"
        full_ids, reference_hash = references[(job["task"], job["sample_id"])]
        selection = json.loads((folder / "selection.json").read_text())
        variants = torch.load(folder / "logits.pt", map_location="cpu", weights_only=True)
        for variant, logits in variants.items():
            ids = logits.argmax(-1).tolist()
            if len(ids) != 128 or len(full_ids) != 128:
                raise ValueError("complete fixed128 model trajectories required")
            record = controls[job["index"]]
            execution_valid = variant == "reference" or (
                record["controls"][variant]["clone_repack_bitwise"]
                and record["controls"][variant].get("lossless_exact_codec_128_bitwise", True)
            )
            storage = (
                record["reference"]["storage"]
                if variant == "reference"
                else record["controls"][variant]["candidate"]["storage"]
            )
            for cap in (32, 64, 128):
                prediction = answer_prefix(ids, eos=eos, cap=cap, decode=decode)
                reference = answer_prefix(full_ids, eos=eos, cap=cap, decode=decode)
                score = rouge_l_f1(prediction["answer"], [reference["answer"]])
                rows.append(
                    dict(
                        run_id=str(root),
                        sample_id=job["sample_id"],
                        task=job["task"],
                        model=model["id"],
                        model_revision=model["revision"],
                        backend="hf_eager",
                        method=job["method"],
                        variant=variant,
                        correction_mode=record.get("correction_mode", "none"),
                        execution_valid=execution_valid,
                        **prediction,
                        implementation_variant=selection.get(
                            "variant", selection.get("selection", job["method"])
                        ),
                        method_label="full_position_lossless_control"
                        if job["method"] == "full_kv" and variant == "reference"
                        else job["method"]
                        + (
                            "_common_lossless_transport"
                            if variant == "reference"
                            else "_prompt_precision_variant"
                        ),
                        rouge_l_fullkv_reference=score,
                        metric_implementation="rouge==1.0.1",
                        reference_artifact_sha256=reference_hash,
                        producer_sha=plan["source_sha"],
                        manifest_path=str(folder / "manifest.json"),
                        measurement_type="retrospective_two_consumed_inputs_model_reference_not_official_task_score",
                        **{
                            k: storage[k]
                            for k in (
                                "initial_active_kv_storage_bytes",
                                "final_active_kv_storage_bytes",
                                "peak_active_kv_storage_bytes",
                            )
                        },
                    )
                )
    output.mkdir()
    pd.DataFrame(rows).to_parquet(output / "observations.parquet", index=False)
    write_once(
        output / "controls.json",
        dict(
            EOS_behavior_separate_from_fixed128_latency=True,
            tasks_not_averaged=True,
            benchmark_annotations_used=False,
            independent_quality_established=False,
            superiority_established=False,
            whole_request_peak_certified=False,
        ),
    )
    write_once(
        output / "manifest.json",
        dict(
            source_sha=clean_source(source),
            parent_producer_sha=plan["source_sha"],
            inputs=inputs,
            outputs={p.name: sha(p) for p in output.iterdir()},
            measurement_type="derived_consumed_model_reference_quality",
        ),
    )
