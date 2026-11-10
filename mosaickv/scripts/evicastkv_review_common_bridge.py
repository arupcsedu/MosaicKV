#!/usr/bin/env python3
"""Consumed BF16 LLaVA common-model bridge; a ceiling is NOT byte matching.

This diagnostic compares the exact selector and shared-source variant under
one checkpoint. It does not port the BF16/full-MHA codec to FP16/GQA Mistral,
certify independent quality, or conflate source and exact-selection designs.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
MODEL = ("llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
ARMS = (
    "fullkv_native_reference",
    "fullkv_HF_forced_control",
    "prompt_attention_topk",
    "mosaickv_exact",
    "source3_attention_optional",
)
POLICY = dict(
    id="fox_BF16_LLaVA1_5_common_model_consumed_bridge_v1",
    model_id=MODEL[0],
    model_revision=MODEL[1],
    precision="bf16",
    attention="eager",
    initial_ceiling_bytes=64 * 2**20,
    output_predictions=128,
    continuation_updates=127,
    mandatory="all_original_text_plus_first4_last8_original_positions",
    mandatory_transport="scoped_evaluation_FullKVState_factory_override_with_pack_check",
    exact_candidate="frozen_MileBench_exact_policy_with_common_BF16_model_and_mandatory",
    source_candidate="existing_source3_attention_optional_finite_protection_ladder",
    conditioning="FullKV_generated_tokens_injected_only_at_decode_boundary",
    selection_reference_access=False,
    labels_access=False,
    source_history="lossless_source_direct",
    exact_history="native_HF_append_only_with_existing_preallocation",
    same_history_storage_representation=False,
    realized_byte_match_claim=False,
    transport_control_absolute_tolerance=0.001,
    guard=dict(tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0),
    seed=13,
    warmups=0,
    repetitions=1,
    no_headline_latency=True,
    no_superiority_claim=True,
    consumed_only=True,
    fresh_confirmation_authorized=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def mandatory_positions(spans, length):
    if type(length) is not int or length < 1:
        raise ValueError("positive original prompt length required")
    text = set()
    for span in spans:
        if not 0 <= span.start <= span.end <= length:
            raise ValueError("modality span outside original prompt")
        if getattr(span.modality, "value", span.modality) == "text":
            text.update(range(span.start, span.end))
    return tuple(sorted(text | set(range(min(4, length))) | set(range(max(0, length - 8), length))))


def exact_config(source_config, template, method):
    """The codec need not be ported: execute exact selectors on its checkpoint."""
    from mosaickv.config import RunConfig

    if method not in ("full_kv", "prompt_attention_topk", "mosaickv_exact"):
        raise ValueError("only registered common-model exact methods")
    if (source_config["model"]["id"], source_config["model"]["revision"]) != MODEL:
        raise ValueError("common bridge must retain the source checkpoint")
    result = deepcopy(template)
    result.update(
        method=method,
        model=deepcopy(source_config["model"]),
        dataset=deepcopy(source_config["dataset"]),
        bounded_decode=None,
    )
    result["model"]["precision"] = "bf16"
    result["cache"].update(
        budget_value=POLICY["initial_ceiling_bytes"],
        block_size=1,
        physical_budget_block_size=1,
        physical_budget_mandatory_policy="method",
        retention_ratio=1.0 if method == "full_kv" else 0.99,
        total_storage_budget_bytes=None,
    )
    result["generation"].update(max_new_tokens=128, do_sample=False, temperature=0.0)
    result["execution"].update(
        seed=POLICY["seed"],
        attention_implementation="eager",
        decode_attention_implementation="eager",
        deterministic_algorithms=True,
    )
    result["prototypes"]["enabled"] = False
    result["residual"]["enabled"] = False
    result["repair"].update(enabled=False, policy="none", max_blocks_per_step=0)
    if method != "mosaickv_exact":
        result["selection"].update(
            routed_head_only_planning=False, stable_head_prompt_routing=False
        )
        for stage in ("forecasting", "graph", "selection"):
            result[stage]["enabled"] = False
    # Preserve the frozen exact graph/forecast policy; simple baselines bypass it.
    return RunConfig.from_mapping(result)


def arm_config(source_config, template, arm):
    """Resolve actual HF/source-driver controls before any model access."""
    from mosaickv.config import RunConfig

    if arm not in ARMS:
        raise ValueError("unregistered bridge arm")
    if arm == ARMS[-1]:
        source = deepcopy(source_config)
        source["execution"].update(
            seed=POLICY["seed"],
            deterministic_algorithms=True,
            attention_implementation="eager",
            decode_attention_implementation="eager",
        )
        source["generation"].update(max_new_tokens=128, do_sample=False, temperature=0.0)
        return RunConfig.from_mapping(source)
    return exact_config(source_config, template, "full_kv" if arm in ARMS[:2] else arm)


def resolved_binding(config, arm):
    from mosaickv.config import canonical_config, config_sha256

    config_sha = config_sha256(config)
    return dict(
        config=canonical_config(config),
        config_sha256=config_sha,
        resolved_execution_sha256=digest(dict(config_sha256=config_sha, arm=arm, policy=POLICY)),
    )


@contextmanager
def aligned_mandatory(facts):
    """Evaluation-only transaction, preserving source K/V and online algorithms."""
    from mosaickv import cache_state
    from mosaickv.backends import hf_runtime

    original = cache_state.FullKVState.from_cache_snapshot
    original_pack = hf_runtime._pack_exact_runtime_blocks

    def build(cls, snapshot, **kwargs):
        positions = mandatory_positions(
            kwargs["modality_spans"], kwargs["original_logical_sequence_length"]
        )
        kwargs["mandatory_logical_positions"] = positions
        result = original(snapshot, **kwargs)
        if result.mandatory_logical_positions != frozenset(positions):
            raise ValueError("common mandatory injection did not reach source partition")
        facts["mandatory_positions"] = positions
        facts["mandatory_sha256"] = digest(positions)
        facts["state_construction_checks"] = facts.get("state_construction_checks", 0) + 1
        return result

    def packed(adapter, full_state, blocks, **kwargs):
        required = set(full_state.mandatory_logical_positions)
        routes = {}
        for block in blocks:
            routes.setdefault((block.layer, block.kv_head), set()).update(
                block.original_logical_positions
            )
        expected_routes = {
            (layer_index, head)
            for layer_index, layer in enumerate(full_state.layers)
            for head in range(layer.kv_heads)
        }
        if set(routes) != expected_routes or any(
            not required <= positions for positions in routes.values()
        ):
            raise ValueError("packed exact cache removed common mandatory evidence")
        facts["all_routes_preserve_common_mandatory"] = True
        return original_pack(adapter, full_state, blocks, **kwargs)

    with (
        patch.object(cache_state.FullKVState, "from_cache_snapshot", classmethod(build)),
        patch.object(hf_runtime, "_pack_exact_runtime_blocks", packed),
    ):
        yield


def exact_request(adapter, payload, config, prefix, trace_directory):
    """Inject reference tokens at the decode boundary, never in selection inputs."""
    import evicastkv_request_validation as messages
    import torch

    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.evaluation.model import EvaluationRequest

    if len(prefix) != 128 or any(type(token) is not int or token < 0 for token in prefix):
        raise ValueError("complete FullKV-generated prefix required")
    values, facts = [], {}
    original_prefill, original_decode = adapter.prefill, adapter.decode_one_token

    def prefill(*args, **kwargs):
        output = original_prefill(*args, **kwargs)
        if values:
            raise ValueError("bridge requires one prefill, no draft rollout")
        values.append(output.logits.detach().reshape(-1).float().cpu())
        return output

    def decode(token, state, *args, **kwargs):
        index = len(values) - 1
        if not 0 <= index < 127:
            raise ValueError("common-model bridge decode exceeds registered horizon")
        forced = torch.full_like(token, prefix[index])
        output = original_decode(forced, state, *args, **kwargs)
        values.append(output.logits.detach().reshape(-1).float().cpu())
        return output

    started = time.monotonic()
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=trace_directory)
    # Deliberately empty generation kwargs: neither reference outputs nor the
    # prefix enter the evaluation request, planner configuration or selector.
    request = EvaluationRequest(
        config.method.value,
        payload["sample_id"],
        payload["task"],
        messages.load_messages(payload),
        {},
    )
    with (
        patch.object(adapter, "prefill", prefill),
        patch.object(adapter, "decode_one_token", decode),
        aligned_mandatory(facts),
    ):
        output = runtime.generate(request)
    if len(values) != 128:
        raise ValueError("incomplete exact bridge trajectory")
    logits = torch.stack(values)
    metrics = dataclasses.asdict(output.metrics)
    initial = metrics["initial_active_kv_storage_bytes"]
    if config.method.value != "full_kv" and (
        type(initial) is not int
        or initial > POLICY["initial_ceiling_bytes"]
        or not facts.get("all_routes_preserve_common_mandatory")
    ):
        raise ValueError("exact bridge cap or mandatory preservation failed")
    traces = list(Path(trace_directory).rglob("*.json"))
    if len(traces) != 1:
        raise ValueError("one complete HF trace required")
    return logits, dict(
        metrics=metrics,
        storage=dict(
            initial_active_kv_storage_bytes=initial,
            final_active_kv_storage_bytes=metrics["final_active_kv_storage_bytes"],
            peak_active_kv_storage_bytes=metrics["peak_active_kv_storage_bytes"],
        ),
        mandatory=facts,
        original_KV_unchanged=True,
        source_variant=False,
        conditioning_prefix_sha256=digest(prefix),
        selector_prefix_access=False,
        free_greedy_sequence=False,
        greedy_predictions_on_FullKV_prefix=True,
        trace_path=str(traces[0]),
        trace_sha256=sha(traces[0]),
        end_to_end_diagnostic_seconds=time.monotonic() - started,
        no_headline_latency=True,
        whole_request_KV_only_peak_certified=False,
    )


def budget_relation(records, cap):
    compressed = [record for record in records if record["arm"] in ARMS[2:]]
    if len(compressed) != 3:
        raise ValueError("all three compressed arms required")
    mandatory_hashes = [
        record.get("mandatory", {}).get("mandatory_sha256") for record in compressed
    ]
    if any(not value for value in mandatory_hashes) or len(set(mandatory_hashes)) != 1:
        raise ValueError("compressed methods did not preserve identical mandatory prompt positions")
    endpoints = {}
    for key in ("initial", "final", "peak"):
        field = f"{key}_active_kv_storage_bytes"
        values = [record["storage"][field] for record in compressed]
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("measured complete storage boundaries required")
        endpoints[key] = dict(values=values, equal=len(set(values)) == 1)
    within = all(
        record["storage"]["initial_active_kv_storage_bytes"] <= cap for record in compressed
    )
    return dict(
        common_initial_ceiling_bytes=cap,
        mandatory_policy_matched=True,
        within_initial_ceiling=within,
        initial_ceiling_shortfalls={
            record["arm"]: cap - record["storage"]["initial_active_kv_storage_bytes"]
            for record in compressed
        },
        endpoints=endpoints,
        all_boundaries_exactly_matched=all(e["equal"] for e in endpoints.values()),
        direct_byte_matched_comparison=False,
        label="common_ceiling_not_matched"
        if not all(e["equal"] for e in endpoints.values())
        else "equal_observed_bytes_but_transport_scope_differs",
    )


def prepare(root, parent):
    from evicastkv_review_source_prefix import _bind_parent

    root, parent = Path(root), Path(parent)
    if root.exists() or not root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch root required")
    revision = clean_source(SOURCE)
    previous, jobs, bindings = _bind_parent(parent)
    template = SOURCE / "mosaickv/configs/experiments/evicastkv_fox/milebench/mosaickv_exact.json"
    for path in (
        template,
        Path(__file__),
        SOURCE / "mosaickv/scripts/evicastkv_review_source_prefix.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_source_controls.py",
        SOURCE / "mosaickv/src/mosaickv/backends/hf_runtime.py",
    ):
        bindings[str(path)] = sha(path)
    resolved = []
    for index, job in enumerate(jobs):
        entries = {}
        for arm in ARMS:
            binding = resolved_binding(
                arm_config(job.get("config", previous["config"]), read(template), arm), arm
            )
            path = root / f"resolved-configs/{index}/{arm}.json"
            write_once(path, binding["config"])
            bindings[str(path)] = sha(path)
            entries[arm] = dict(
                path=str(path),
                config_sha256=binding["config_sha256"],
                resolved_execution_sha256=binding["resolved_execution_sha256"],
            )
        resolved.append(entries)
    environment_lock = SOURCE / "mosaickv/env/common/requirements.lock"
    bindings[str(environment_lock)] = sha(environment_lock)
    plan = dict(
        source_sha=revision,
        parent=str(parent),
        jobs=jobs,
        config=previous["config"],
        template=str(template),
        bindings=bindings,
        resolved_configs=resolved,
        environment_lock_path=str(environment_lock),
        environment_lock_sha256=sha(environment_lock),
        arms=ARMS,
        policy=POLICY,
        expected_rows=10,
        maximum_workers=2,
        maximum_a100s_total=6,
        cpus_per_worker=12,
        time_limit_seconds=7200,
        fresh_confirmation_authorized=False,
        shared_realized_byte_comparison=False,
        paper_superiority_claim=False,
    )
    write_once(root / "plan.json", plan)
    return dict(workers=2, expected_rows=10, plan_sha256=sha(root / "plan.json"))


def validate(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy"] != POLICY or tuple(plan["arms"]) != ARMS:
        raise ValueError("common-model policy drift")
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"common-model source/input changed: {path}")
    return plan


def worker(root, index):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_review_source_prefix import (
        comparisons,
        forced_prefix,
        label_blind_payload,
        native_reference,
    )
    from evicastkv_review_source_prefix import request as source_request

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = validate(root)
    if type(index) is not int or index not in range(2):
        raise ValueError("only two consumed inputs permitted")
    output = root / f"workers/{index}"
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], index=index)
    rows, saved = [], {}
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one A100 and deterministic prelaunch workspace required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(POLICY["seed"])
        job = plan["jobs"][index]
        payload = label_blind_payload(job["inputs"][0])
        original_config = job.get("config", plan["config"])
        base = arm_config(original_config, read(plan["template"]), ARMS[-1])
        if (base.model.id, base.model.revision) != MODEL or base.model.precision.value != "bf16":
            raise ValueError("pinned source BF16 checkpoint required")
        write_once(
            output / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=sha(root / "plan.json"),
                model=dataclasses.asdict(base.model),
                dataset=dataclasses.asdict(base.dataset),
                processor_revision=MODEL[1],
                environment=environment(),
                gpu=dataclasses.asdict(capture_gpu_environment(torch)),
                backend="huggingface",
                attention="eager",
                seed=POLICY["seed"],
                measurement_type="consumed_common_model_bridge_diagnostic",
                result_category="unified_comparison",
                result_origin="locally_measured",
                environment_lock_path=plan["environment_lock_path"],
                environment_lock_sha256=plan["environment_lock_sha256"],
                cuda_runtime=torch.version.cuda,
                vllm="not_used",
                sglang="not_used",
                resolved_configs=plan["resolved_configs"][index],
                sample_id=payload["sample_id"],
                input_sha256=sha(Path(job["inputs"][0])),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                labels_available=False,
            ),
        )
        adapter = load_hf_adapter(
            MODEL[0],
            revision=MODEL[1],
            model_kwargs=_model_kwargs(base, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        with torch.inference_mode(), (output / "raw.jsonl").open("x") as stream:
            reference = prefix = None
            for arm in ARMS:
                from mosaickv.config import config_sha256

                binding = plan["resolved_configs"][index][arm]
                current = RunConfig.from_mapping(read(binding["path"]))
                if config_sha256(current) != binding["config_sha256"] or (
                    resolved_binding(current, arm)["resolved_execution_sha256"]
                    != binding["resolved_execution_sha256"]
                ):
                    raise ValueError("actual resolved arm configuration differs from registration")
                row = dict(
                    arm=arm,
                    status="failed",
                    sample_id=payload["sample_id"],
                    manifest_path=str(output / "manifest.json"),
                    manifest_sha256=sha(output / "manifest.json"),
                    config_sha256=config_sha256(current),
                    resolved_config_path=binding["path"],
                    resolved_execution_sha256=binding["resolved_execution_sha256"],
                    result_category="unified_comparison",
                    result_origin="locally_measured",
                    model_id=MODEL[0],
                    model_revision=MODEL[1],
                    precision="bf16",
                    same_initial_ceiling_not_realized_byte_matching=True,
                )
                try:
                    gc.collect()
                    torch.cuda.synchronize()
                    if arm == ARMS[0]:
                        logits, record = native_reference(adapter, payload)
                        reference, prefix = logits.clone(), forced_prefix(logits)
                    elif arm == ARMS[-1]:
                        logits, record = source_request(
                            adapter,
                            payload,
                            current,
                            arm,
                            prefix=prefix,
                            cap=POLICY["initial_ceiling_bytes"],
                        )
                        protections = record["protection_records"]
                        if (
                            not protections
                            or len({digest(p["mandatory_positions"]) for p in protections}) != 1
                        ):
                            raise ValueError("source layers disagree on mandatory positions")
                        record["mandatory"] = dict(
                            mandatory_positions=protections[0]["mandatory_positions"],
                            mandatory_sha256=digest(protections[0]["mandatory_positions"]),
                            all_routes_preserve_common_mandatory=True,
                        )
                    else:
                        logits, record = exact_request(
                            adapter, payload, current, prefix, output / "traces" / arm
                        )
                    row.update(record, numerical=comparisons(reference, logits))
                    if arm == ARMS[1] and (
                        row["numerical"]["prefix_128"]["token_agreement"] != 1.0
                        or row["numerical"]["prefix_128"]["maximum_logit_difference"]
                        > POLICY["transport_control_absolute_tolerance"]
                    ):
                        raise RuntimeError("BF16 HF FullKV transport parity failed")
                    if arm == ARMS[-1] and record.get("physical_allocation", {}).get(
                        "uncapped_numerical_diagnostic"
                    ):
                        raise RuntimeError("source packet cannot fit registered ceiling")
                    saved[arm] = logits.detach().cpu()
                    row["status"] = "completed"
                except Exception as exception:
                    row["error"] = f"{type(exception).__name__}: {exception}"
                    result["failures"].append(dict(arm=arm, error=row["error"]))
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                if arm in ARMS[:2] and row["status"] != "completed":
                    break
        with (output / "logits.pt").open("xb") as destination:
            torch.save(saved, destination)
        if not result["failures"] and len(rows) == len(ARMS):
            result.update(
                status="completed",
                budget_relation=budget_relation(rows, POLICY["initial_ceiling_bytes"]),
            )
    except Exception as exception:
        result["failures"].append(dict(error=f"{type(exception).__name__}: {exception}"))
    result["rows"] = len(rows)
    result["outputs"] = {path.name: sha(path) for path in output.iterdir() if path.is_file()}
    write_once(output / "result.json", result)
    return result


def analyze(root):
    root = Path(root)
    plan = validate(root)
    failures, workers, rows = [], [], []
    for index in range(2):
        path = root / f"workers/{index}/result.json"
        try:
            result = read(path)
            workers.append(result)
            if (
                result["source_sha"] != plan["source_sha"]
                or result["status"] != "completed"
                or result["failures"]
                or result["rows"] != 5
            ):
                raise ValueError("incomplete common-model worker")
            for name, expected in result["outputs"].items():
                if sha(path.parent / name) != expected:
                    raise ValueError("worker artifact changed")
            current = [
                json.loads(line) for line in (path.parent / "raw.jsonl").read_text().splitlines()
            ]
            if tuple(row["arm"] for row in current) != ARMS or any(
                row["status"] != "completed" for row in current
            ):
                raise ValueError("incomplete or reordered registered arms")
            for row in current:
                expected = plan["resolved_configs"][index][row["arm"]]
                if (
                    row.get("config_sha256") != expected["config_sha256"]
                    or (
                        row.get("resolved_execution_sha256")
                        != expected["resolved_execution_sha256"]
                    )
                    or row.get("resolved_config_path") != expected["path"]
                ):
                    raise ValueError("raw row does not bind its actual resolved arm configuration")
                if row.get("manifest_sha256") != sha(Path(row["manifest_path"])):
                    raise ValueError("raw row manifest binding changed")
            if not budget_relation(current, POLICY["initial_ceiling_bytes"])[
                "within_initial_ceiling"
            ]:
                raise ValueError("common-model initial ceiling exceeded")
            rows.extend(current)
        except Exception as exception:
            failures.append(dict(index=index, error=f"{type(exception).__name__}: {exception}"))
    gate = dict(
        status="passed_execution" if not failures and len(rows) == 10 else "failed_execution",
        source_sha=plan["source_sha"],
        rows=len(rows),
        failures=failures,
        transport_controls_passed=not failures and len(rows) == 10,
        source_and_exact_share_model=True,
        shared_realized_byte_comparison=False,
        history_representation_differs=True,
        independent_quality_evidence=False,
        fresh_confirmation_authorized=False,
        superiority_claim=False,
        fixed_prefix_diagnostic_not_free_generation=True,
    )
    write_once(root / "analysis/rows.json", rows)
    write_once(root / "analysis/GATE_common_bridge.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare" and args.parent is None:
        parser.error("prepare requires --parent")
    result = (
        prepare(args.root, args.parent)
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
