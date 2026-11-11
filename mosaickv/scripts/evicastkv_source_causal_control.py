#!/usr/bin/env python3
"""Consumed-input prompt-K/V causal controls with identical source-history policy."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import evicastkv_review_source_controls as source
import evicastkv_review_source_prefix as fixed

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/audits/fox-review-v7-closure-20261110/source-prefix")
RECOVERY = Path(
    "/scratch/djy8hg/audits/fox-review-v7-source-analysis-recovery-20261110/source-prefix"
)
ARMS = {
    "raw_prompt_K_raw_prompt_V": (True, True),
    "source_prompt_K_raw_prompt_V": (False, True),
    "raw_prompt_K_source_prompt_V": (True, False),
    "source_prompt_K_source_prompt_V": (False, False),
}
POLICY = dict(
    id="consumed_prompt_KV_factorial_identical_lossless_source_history_v1",
    source_protection="unchanged_V7_attention_optional_attainable_ladder",
    history="identical_existing_direct_lossless_source_history_policy_not_identical_hidden_values",
    output_predictions=128,
    decode_updates=127,
    conditioning="frozen_V7_FullKV_prefix_not_benchmark_annotations",
    selector_prefix_access=False,
    guard=fixed.POLICY["guard"],
    repeat_native_and_raw_control_logit_atol=0.001,
    source_replay=False,
    all_arms_reconstruct_source_before_prompt_only_counterfactual_copy=True,
    all_arms_retain_and_charge_both_original_prompt_components=True,
    original_source_retirement_claim=False,
    raw_components_checked_at_first_and_last_fill=True,
    trials=1,
    warmups=0,
    soft_worker_seconds=1100,
    hard_worker_seconds=1200,
    independent_quality_N=0,
    no_fresh_cohort=True,
    no_quality_or_latency_claim=True,
    no_shared_realized_byte_claim=True,
    no_whole_request_KV_peak_certificate=True,
)


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
        raise ValueError("finite original tensor required")
    value = hashlib.sha256()
    value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
    value.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return value.hexdigest()


def owner(tensor):
    return str(tensor.device), tensor.untyped_storage().data_ptr()


@dataclass
class PromptCounterfactual:
    """Both originals are declared owners in every arm; only destinations change."""

    packet: object
    original: tuple
    raw_components: tuple
    fill_calls: int = 0
    verified_raw_copies: list = field(default_factory=lambda: [0, 0])
    kind: str = "evicastkv_consumed_prompt_component_causal_counterfactual"

    def __post_init__(self):
        import torch

        if (
            len(self.original) != 2
            or len(self.raw_components) != 2
            or any(type(v) is not bool for v in self.raw_components)
            or any(
                tuple(t.shape) != tuple(self.packet.shape)
                or t.ndim != 4
                or t.shape[0] != 1
                or t.dtype != torch.bfloat16
                or t.device != self.original[0].device
                for t in self.original
            )
        ):
            raise ValueError(
                "complete batch-one original BF16 K/V and explicit component flags required"
            )

    @property
    def shape(self):
        return self.packet.shape

    def private(self):
        return (*self.packet.private(), *self.original)

    def shared(self):
        return self.packet.shared()

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

        return backing_bytes(self.tensors())

    def fill(self, attention, destinations, *, direct):
        import torch

        n = self.shape[-2]
        if (
            not direct
            or len(destinations) != 2
            or any(
                t.dtype != self.original[0].dtype
                or t.device != self.original[0].device
                or t.ndim != 4
                or tuple(t.shape[:2]) != tuple(self.shape[:2])
                or t.shape[-1] != self.shape[-1]
                or t.shape[-2] < n
                or owner(t) in {owner(v) for v in self.tensors()}
                for t in destinations
            )
            or owner(destinations[0]) == owner(destinations[1])
        ):
            raise ValueError("private disjoint complete K/V destinations required")
        self.packet.fill(attention, destinations, direct=True)
        self.fill_calls += 1
        for index, (use_raw, original, target) in enumerate(
            zip(self.raw_components, self.original, destinations, strict=True)
        ):
            if use_raw:
                prompt = target[..., :n, :]
                prompt.copy_(original)
                if self.fill_calls in (1, 127):
                    if not torch.equal(prompt.view(torch.int16), original.view(torch.int16)):
                        raise RuntimeError("original prompt component was not restored bitwise")
                    self.verified_raw_copies[index] += 1


def prepare(root):
    root = Path(root)
    revision = clean_source(SOURCE)
    plan_path = PARENT / "plan.json"
    recovery_path = RECOVERY / "analysis/GATE_source_fixed_prefix.json"
    plan, gate = read(plan_path), read(recovery_path)
    if (
        gate.get("complete") is not True
        or gate.get("failures") != []
        or gate.get("actual_rows") != 14
        or gate.get("plan_sha256") != sha(plan_path)
        or len(plan["jobs"]) != 2
    ):
        raise ValueError("completed two-input fixed-prefix evidence required")
    jobs, files = [], [plan_path, recovery_path]
    for index, job in enumerate(plan["jobs"]):
        payload = fixed.label_blind_payload(job["inputs"][0])
        if sha(Path(job["inputs"][0])) != job["input_sha256"]:
            raise ValueError("consumed input binding changed")
        config = json.loads(json.dumps(job.get("config", plan["config"])))
        config["execution"]["seed"] = 13
        jobs.append(dict(job, config=config, reference_worker=index))
        files.append(Path(job["inputs"][0]))
        for item in payload["media"]:
            path = Path(item["path"])
            if sha(path) != item["sha256"]:
                raise ValueError("consumed media binding changed")
            files.append(path)
        worker_path = PARENT / f"workers/{index}"
        old = read(worker_path / "result.json")
        if old["status"] != "completed" or old["failures"]:
            raise ValueError("saved source producer incomplete")
        files.append(worker_path / "result.json")
        for filename, expected in old["outputs"].items():
            path = worker_path / filename
            if sha(path) != expected:
                raise ValueError("saved source evidence changed")
            files.append(path)
    files += [Path(__file__), Path(source.__file__), Path(fixed.__file__)]
    files += [
        SOURCE / f"mosaickv/src/mosaickv/evaluation/{file}.py"
        for file in ("storage_runs", "shared_source_format", "graphed_storage", "source_history")
    ]
    files.append(SOURCE / "mosaickv/env/common/requirements.lock")
    write_once(
        root / "protocol.json",
        dict(
            source_sha=revision,
            policy=POLICY,
            policy_sha256=digest(POLICY),
            arms=ARMS,
            parent=str(PARENT),
            jobs=jobs,
            bindings={str(p): sha(p) for p in files},
            expected_rows=8,
            workers=1,
            no_fresh_cohort=True,
        ),
    )
    return dict(workers=1, expected_rows=8, protocol_sha256=sha(root / "protocol.json"))


def validate(root):
    plan = read(Path(root) / "protocol.json")
    clean_source(SOURCE, plan["source_sha"])
    if (
        digest(plan["policy"]) != digest(POLICY)
        or plan["policy_sha256"] != digest(POLICY)
        or digest(plan["arms"]) != digest(ARMS)
        or any(sha(Path(p)) != h for p, h in plan["bindings"].items())
    ):
        raise ValueError("immutable causal-control protocol changed")
    return plan


def request(adapter, payload, arm, *, prefix, source_cap):
    """Prefix is consumed by decoding only, not source construction or protection."""
    import evicastkv_request_validation as messages
    import torch

    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.evaluation.storage_runs import run

    if arm not in ARMS or len(prefix) != 128 or any(type(t) is not int or t < 0 for t in prefix):
        raise ValueError(
            "registered prompt component intervention and complete frozen prefix required"
        )
    phases, allocation = {}, {}

    def timed(name, action):
        torch.cuda.synchronize()
        begin = time.monotonic()
        result = action()
        torch.cuda.synchronize()
        phases[name] = time.monotonic() - begin
        return result

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    prepared = timed(
        "media_processor", lambda: adapter.prepare_inputs(messages.load_messages(payload))
    )
    with (
        PromptQueryCapture(adapter, summaries=True, normalized=False, prior=False) as attention,
        SourceCapture(adapter) as normalized,
        PhaseCapture(adapter) as rotary,
    ):
        prefill = timed(
            "captured_prefill", lambda: adapter.prefill(prepared, capture_queries=False)
        )
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    if len(pairs) != 32 or any(
        tuple(t.shape[:2]) != (1, 32) or t.shape[-1] != 128 for p in pairs for t in p
    ):
        raise ValueError("source control requires pinned full-MHA LLaVA geometry")
    packets, _ = timed(
        "unchanged_source_construction",
        lambda: fixed._formats_with_ceiling(
            adapter,
            prepared,
            prefill,
            attention,
            normalized,
            rotary,
            "source3_attention_optional",
            source_cap,
            allocation,
        ),
    )
    source_hash = source._packet_hash(packets)
    original_hash = digest([[tensor_sha(t) for t in p] for p in pairs])
    raw_bytes = backing_bytes(tuple(t for p in pairs for t in p))
    wrappers = tuple(
        PromptCounterfactual(packet, tuple(pair), ARMS[arm])
        for packet, pair in zip(packets, pairs, strict=True)
    )
    charged = backing_bytes(tuple(t for p in wrappers for t in p.tensors()))
    if charged < raw_bytes or any(
        not set(map(owner, pair)) <= set(map(owner, wrapper.tensors()))
        for pair, wrapper in zip(pairs, wrappers, strict=True)
    ):
        raise ValueError("retained original cache backing was not fully declared")

    def retire_nonowners():
        prefill.state.past_key_values = None
        prepared.model_inputs.clear()
        attention.records.clear()
        attention.windows.clear()
        attention.priors.clear()
        normalized.records.clear()
        rotary.phases = None
        gc.collect()

    timed("retire_nonowned_captures_not_original_KV", retire_nonowners)
    logits, record = timed(
        "decode_counterfactual_including_instrumentation",
        lambda: run(
            adapter,
            prefill,
            wrappers,
            tokens=128,
            forced=prefix,
            audit=False,
            source_history=True,
            source_history_constructor="direct",
            shared_workspace=True,
            workspace_graph=False,
        ),
    )
    if source_hash != source._packet_hash(packets) or original_hash != digest(
        [[tensor_sha(t) for t in p] for p in pairs]
    ):
        raise RuntimeError("counterfactual mutated source packets or original K/V")
    expected_copies = [2 if raw else 0 for raw in ARMS[arm]]
    if (
        any(w.fill_calls != 127 or w.verified_raw_copies != expected_copies for w in wrappers)
        or record["used_fullkv_fallback"]
        or record["storage"]["history_updates_per_layer"] != [127] * 32
        or record["storage"]["initial_active_kv_storage_bytes"] < charged
        or record["storage"]["active_workspace_bytes"]
    ):
        raise RuntimeError(
            "component replacement, shared-history isolation or storage charge failed"
        )
    record.update(
        source_representation_sha256=source_hash,
        retained_original_KV_sha256=original_hash,
        original_raw_KV_bytes=raw_bytes,
        counterfactual_declared_prompt_owner_bytes=charged,
        retained_original_KV_fully_charged=True,
        original_KV_retired=False,
        source_allocation=allocation,
        phase_wall_seconds=phases,
        raw_component_flags=ARMS[arm],
        fill_calls_per_layer=[w.fill_calls for w in wrappers],
        raw_copy_verifications_per_layer=[w.verified_raw_copies for w in wrappers],
        reference_prefix_sha256=digest(prefix),
        selector_prefix_access=False,
        no_shared_byte_or_quality_claim=True,
        whole_request_KV_only_peak_certified=False,
        whole_request_allocator_peak_bytes=int(torch.cuda.max_memory_allocated()),
        whole_request_allocator_reserved_peak_bytes=int(torch.cuda.max_memory_reserved()),
    )
    return logits, record


def worker(root, index=0):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = validate(root)
    if type(index) is not int or index != 0:
        raise ValueError("one registered causal-control worker only")
    output = root / "worker"
    output.mkdir(parents=True, exist_ok=False)
    failures, saved, controls, completed = [], {}, [], 0
    deadline = time.monotonic() + POLICY["soft_worker_seconds"]
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one deterministic A100 required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        cfg = RunConfig.from_mapping(plan["jobs"][0]["config"])
        if (
            (cfg.model.id, cfg.model.revision) != source.MODEL
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
        ):
            raise ValueError("only the pinned BF16 full-MHA eager source checkpoint is supported")
        adapter = load_hf_adapter(
            source.MODEL[0],
            revision=source.MODEL[1],
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        write_once(
            output / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                policy_sha256=digest(POLICY),
                protocol_sha256=sha(root / "protocol.json"),
                model=asdict(cfg.model),
                processor_revision=source.MODEL[1],
                environment=environment(),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                gpu=asdict(capture_gpu_environment(torch)),
                cuda_runtime=torch.version.cuda,
                vllm="not_used",
                sglang="not_used",
                backend="huggingface",
                attention="eager",
                seed=13,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                measurement_type="consumed_prompt_component_causal_counterfactual",
                result_category="unified_comparison",
                result_origin="locally_measured",
                benchmark_labels_read=False,
                independent_quality_N=0,
                config_sha256=[
                    config_sha256(RunConfig.from_mapping(j["config"])) for j in plan["jobs"]
                ],
                datasets=[j["config"]["dataset"] for j in plan["jobs"]],
            ),
        )
        with (output / "raw.jsonl").open("x") as stream, torch.inference_mode():
            for job_index, job in enumerate(plan["jobs"]):
                payload = fixed.label_blind_payload(job["inputs"][0])
                config = RunConfig.from_mapping(job["config"])
                old = PARENT / f"workers/{job['reference_worker']}"
                reference = torch.load(old / "logits.pt", map_location="cpu", weights_only=True)[
                    "fullkv_native_reference"
                ]
                prefix = fixed.forced_prefix(reference)
                frozen = read(old / "reference_prefix.json")
                if frozen["tokens"] != list(prefix) or frozen["sha256"] != digest(prefix):
                    raise ValueError("frozen FullKV prefix binding mismatch")
                if time.monotonic() >= deadline:
                    raise TimeoutError("bounded causal-control soft deadline reached")
                repeated, _ = fixed.native_reference(adapter, payload)
                native = fixed.comparisons(reference, repeated)
                controls.append(dict(sample_id=payload["sample_id"], diagnostics=native))
                if (
                    native["prefix_128"]["token_agreement"] != 1
                    or native["prefix_128"]["maximum_logit_difference"] > 0.001
                ):
                    raise ValueError("native control differs from frozen reference")
                for arm in ARMS:
                    row = dict(
                        sample_id=payload["sample_id"],
                        task=payload["task"],
                        input_index=job_index,
                        arm=arm,
                        status="failed",
                        manifest_path=str(output / "manifest.json"),
                        config_sha256=config_sha256(config),
                        dataset=asdict(config.dataset),
                        reference_artifact_sha256=sha(old / "logits.pt"),
                        source_sha=plan["source_sha"],
                    )
                    try:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("bounded causal-control soft deadline reached")
                        logits, record = request(
                            adapter,
                            payload,
                            arm,
                            prefix=prefix,
                            source_cap=job["initial_ceiling_bytes"],
                        )
                        numerical = fixed.comparisons(reference, logits)
                        row.update(record, diagnostics=numerical, status="completed")
                        saved[f"{job_index}|{arm}"] = logits
                        if arm == "raw_prompt_K_raw_prompt_V" and (
                            numerical["prefix_128"]["token_agreement"] != 1
                            or numerical["prefix_128"]["maximum_logit_difference"] > 0.001
                        ):
                            raise ValueError(
                                "raw/raw prompt with common lossless history failed clone control"
                            )
                        completed += 1
                    except Exception as error:
                        row.update(status="failed", error=f"{type(error).__name__}: {error}")
                        failures.append(dict(input_index=job_index, arm=arm, error=row["error"]))
                    stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")
                    stream.flush()
        torch.save(saved, output / "logits.pt")
    except Exception as error:
        failures.append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        write_once(output / "native_controls.json", controls)
        write_once(
            output / "result.json",
            dict(
                source_sha=plan["source_sha"],
                protocol_sha256=sha(root / "protocol.json"),
                status="completed" if not failures else "failed",
                failures=failures,
                completed_rows=completed,
                outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()},
            ),
        )


def analyze(root):
    import torch

    root = Path(root)
    plan = validate(root)
    fixed.analysis_cpu_controls()
    failures, rows, summaries = [], [], []
    try:
        result = read(root / "worker/result.json")
        if (
            result["source_sha"] != plan["source_sha"]
            or result["protocol_sha256"] != sha(root / "protocol.json")
            or any(sha(root / "worker" / p) != h for p, h in result["outputs"].items())
        ):
            raise ValueError("causal-control worker artifacts changed")
        failures.extend(result["failures"])
        rows = [json.loads(s) for s in (root / "worker/raw.jsonl").read_text().splitlines()]
        saved = torch.load(root / "worker/logits.pt", map_location="cpu", weights_only=True)
        if (
            len(rows) != 8
            or len({(r["input_index"], r["arm"]) for r in rows}) != 8
            or any(r["status"] != "completed" for r in rows)
        ):
            raise ValueError("missing, duplicate or failed causal-control rows")
        for index, job in enumerate(plan["jobs"]):
            group = [r for r in rows if r["input_index"] == index]
            if {r["arm"] for r in group} != set(ARMS) or any(
                len({r[k] for r in group}) != 1
                for k in (
                    "source_representation_sha256",
                    "retained_original_KV_sha256",
                    "reference_prefix_sha256",
                )
            ):
                raise ValueError(
                    "source packets, original K/V or conditioning differ across interventions"
                )
            reference = torch.load(
                PARENT / f"workers/{job['reference_worker']}/logits.pt",
                map_location="cpu",
                weights_only=True,
            )["fullkv_native_reference"]
            for row in group:
                logits = saved[f"{index}|{row['arm']}"]
                fixed.verify_recomputed_diagnostics(
                    fixed.comparisons(reference, logits), row["diagnostics"]
                )
                summaries.append(
                    dict(
                        input_index=index,
                        arm=row["arm"],
                        diagnostics=row["diagnostics"],
                        storage=row["storage"],
                        original_raw_KV_bytes=row["original_raw_KV_bytes"],
                    )
                )
    except Exception as error:
        failures.append(dict(error=f"{type(error).__name__}: {error}"))
    write_once(root / "causal_summary.json", dict(rows=summaries, policy=POLICY))
    gate = dict(
        status="passed_causal_capture" if not failures else "blocked",
        failures=failures,
        rows=len(rows),
        summary_sha256=sha(root / "causal_summary.json"),
        source_sha=plan["source_sha"],
        no_shared_byte_comparison=True,
        retained_original_KV_counterfactual_not_deployable=True,
        fresh_confirmation_authorized=False,
        independent_quality_N=0,
        superiority_established=False,
        whole_request_KV_peak_certified=False,
        headline_latency_eligible=False,
        all_lossy_arms_pass_guard=len(summaries) == 8
        and all(
            s["diagnostics"]["registered_guard_32"]["registered_guard_pass"] for s in summaries
        ),
    )
    write_once(root / "GATE_source_causal_control.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](args.root), indent=2))


if __name__ == "__main__":
    main()
