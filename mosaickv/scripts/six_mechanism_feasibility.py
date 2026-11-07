#!/usr/bin/env python3
"""Finite canary -> analysis -> conditional component study; no storage-variant edits."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import traceback
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from mosaickv.config import canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import digest, sha, write_once
from mosaickv.evaluation.six_mechanism_gate import all_pass, build_config, validate_evidence

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "mosaickv/configs/experiments/six_mechanism_feasibility_v1.json"
SHELL = SOURCE / "mosaickv/slurm/six_mechanism_feasibility.sbatch"
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def clean_source(expected=None):
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True)
    if status:
        raise ValueError(
            "six-mechanism GPU execution requires a clean separate worktree: " + status
        )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if expected is not None and actual != expected:
        raise ValueError("source revision changed")
    return actual


def checked(root, expected):
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("package import does not belong to the immutable execution worktree")
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable plan changed")
    plan = read(root / "plan.json")
    clean_source(plan["source_sha"])
    if (
        sha(PROTOCOL) != plan["protocol_sha256"]
        or sha(root / "inputs.json") != plan["inputs_sha256"]
    ):
        raise ValueError("frozen protocol/input content changed")
    if not read(root / "cpu.json")["passed"]:
        raise ValueError("CPU feasibility must pass before GPU submission")
    return plan


def submit(root, action, *, array=None, after=None, phase="canary"):
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if array is not None:
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            f"--array=0-{array - 1}%2",
            "--hold",
            "--time=00:30:00",
        ]
    else:
        args += [
            "--partition=bii",
            "--cpus-per-task=2",
            "--mem=8G",
            "--time=01:00:00" if action == "controller" else "--time=00:10:00",
        ]
    if after:
        args += ["--dependency=afterany:" + str(after)]
    log = root / "logs"
    log.mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-six-{phase}-{action}",
        f"--output={log}/{phase}-{action}-%A_%a.out",
        f"--error={log}/{phase}-{action}-%A_%a.err",
        str(SHELL),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
        phase,
    ]
    return subprocess.check_output(args, text=True).strip()


def launch(root, phase):
    array = submit(root, "worker", array=2, phase=phase)
    policy = dict(
        timezone="America/New_York",
        window="all_times",
        maximum_worker_minutes=120,
        maximum_a100s=4,
        arrays=[array],
        adopt_existing_holds=[array],
    )
    write_once(root / f"{phase}_admission.json", policy)
    controller = submit(root, "controller", phase=phase)
    analyzer = submit(root, "analyze", after=array, phase=phase)
    receipt = dict(
        array=array,
        controller=controller,
        analyzer=analyzer,
        gpu_count_per_worker=1,
        cpus_per_worker=12,
        array_maximum_a100s=2,
        worker_limit_minutes=30,
    )
    write_once(root / f"{phase}_submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def prepare(root):
    source_sha = clean_source()
    protocol = read(PROTOCOL)
    parent = Path(protocol["parent_ledger"])
    if sha(parent) != protocol["parent_ledger_sha256"]:
        raise ValueError("reviewed consumed ledger hash mismatch")
    parent_data = read(parent)
    chosen = protocol["consumed_sample_indices"] + protocol["component_sample_indices"]
    inputs = {}
    for index in chosen:
        sample = parent_data["samples"][index]
        # Deliberately exclude references from the model-facing artifact.
        item = {key: sample[key] for key in ("sample_id", "prompt", "media")}
        for media in item["media"]:
            if media["kind"] != "image" or sha(Path(media["path"])) != media["sha256"]:
                raise ValueError("consumed media unavailable or modified")
        inputs[str(index)] = item
    root.mkdir(parents=True, exist_ok=False)
    write_once(root / "inputs.json", inputs)
    tests = [
        "test_six_mechanism_gate",
        "test_forecasting",
        "test_prototype_rope",
        "test_repair",
        "test_graph",
        "test_selection",
        "test_three_tier_cache",
        "test_hf_runtime",
    ]
    with (root / "cpu_tests.txt").open("x") as output:
        completed = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                *[f"mosaickv/tests/unit/{test}.py" for test in tests],
                "mosaickv/tests/integration/test_six_mechanism_runtime.py",
            ],
            cwd=SOURCE,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    write_once(
        root / "cpu.json",
        dict(passed=completed.returncode == 0, tests_sha256=sha(root / "cpu_tests.txt")),
    )
    configs = {
        arm: canonical_config(build_config(protocol, arm))
        for arm in set(protocol["control_arms"] + protocol["component_arms"])
    }
    write_once(
        root / "plan.json",
        dict(
            source_sha=source_sha,
            protocol=protocol,
            protocol_sha256=sha(PROTOCOL),
            inputs_sha256=sha(root / "inputs.json"),
            configs=configs,
            classification=protocol["classification"],
            immutable_storage_parent="a6f689f",
            large_campaign_authorized=False,
            benchmark_labels_available_to_model=False,
        ),
    )
    if completed.returncode:
        raise RuntimeError("CPU proof failed; no GPU jobs submitted")
    launch(root, "canary")


def execute_arm(adapter, config, sample, folder, arm, sequence):
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.six_mechanism_probe import observe_runtime, tensor_sha
    from mosaickv.manifest import ArtifactProvenance, InputProvenance, RunManifestWriter
    from mosaickv.types import MeasurementType

    run_id = f"six-{folder.parent.name}-{folder.name}-{sequence}-{arm}"
    messages = build_multimodal_messages(
        sample["prompt"],
        tuple(MediaItem(MediaKind.IMAGE, media["path"]) for media in sample["media"]),
    )
    request = EvaluationRequest(
        run_id,
        sample["sample_id"],
        "consumed_coco_mechanism_diagnostic",
        messages,
        {"_mosaickv_capture_kv_ownership": True},
    )
    record = dict(
        run_id=run_id,
        arm=arm,
        sample_id=sample["sample_id"],
        config_sha256=config_sha256(config),
        status="started",
        started_at_utc=datetime.now(UTC).isoformat(),
        manifest_path=str(folder / f"{sequence:02}-{arm}.manifest.json"),
    )
    tokenization_sha = digest(dict(prompt=sample["prompt"], status="preparation_not_completed"))
    preprocessing_sha = digest(dict(model=config.model.id, revision=config.model.revision))
    try:
        prepared = adapter.prepare_inputs(messages)
        tokenization_sha = tensor_sha(prepared.model_inputs["input_ids"])
        preprocessing_sha = digest(
            {
                key: tensor_sha(value)
                for key, value in prepared.model_inputs.items()
                if hasattr(value, "detach")
            }
        )
        with observe_runtime(adapter, config) as operations:
            generation = HuggingFaceMosaicKVModel(
                adapter, config, trace_directory=folder / "traces"
            ).generate(request)
        paths = list((folder / "traces" / run_id).glob("*.json"))
        if len(paths) != 1:
            raise ValueError("exactly one complete trace required")
        trace = read(paths[0])
        record.update(
            status="completed",
            answer=generation.answer,
            metrics=asdict(generation.metrics),
            effective_method=generation.effective_method,
            operations=operations,
            trace_path=str(paths[0]),
            trace_sha256=sha(paths[0]),
            tokens=trace["generated_token_ids"],
            storage=trace["cache_lifecycle"],
            residual_bytes=trace["residual_kv_bytes"],
            registered_kv_ownership=trace.get("request_KV_ownership"),
            checks=(
                validate_evidence(trace, operations, controlled=arm == "controlled_repair")
                if arm in {"full_mechanisms", "controlled_repair"}
                else {}
            ),
        )
    except Exception as error:
        record.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    raw = folder / f"{sequence:02}-{arm}.json"
    write_once(raw, record)
    media_sha = digest([item["sha256"] for item in sample["media"]])
    provenance = InputProvenance(
        digest(sample["prompt"]), media_sha, preprocessing_sha, tokenization_sha
    )
    manifest = folder / f"{sequence:02}-{arm}.manifest.json"
    manifest_record = RunManifestWriter(SOURCE).build(
        config,
        MeasurementType.VALIDATION_SMOKE,
        provenance,
        ArtifactProvenance(sha(raw), "not_applicable", sha(raw)),
        run_id=run_id,
        started_at_utc=record["started_at_utc"],
        execution_metadata=dict(
            diagnostic_only=True,
            controlled_repair=arm == "controlled_repair",
            measured_latency_is_instrumented=True,
            prefill_attention_capture="full_original_eager_no_window_kernel_replacement",
            slurm_job_id=os.environ.get("SLURM_JOB_ID", "not_used"),
        ),
    )
    # nvidia-smi can enumerate the entire shared node despite a single visible
    # CUDA device. Preserve that observation separately from allocated count.
    import torch

    manifest_record["hardware"]["node_gpu_count_from_nvidia_smi"] = manifest_record["hardware"][
        "gpu_count"
    ]
    manifest_record["hardware"]["gpu_count"] = torch.cuda.device_count()
    manifest_record["hardware"]["gpu_type"] = torch.cuda.get_device_name()
    write_once(manifest, manifest_record)
    return record


def worker(root, expected, phase, index):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import compare_with_generate
    from mosaickv.backends.hf_runtime import compare_runtime_retention_one
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    protocol = plan["protocol"]
    if (
        phase == "component"
        and not read(root / "canary_analysis/decision.json")["all_gates_passed"]
    ):
        raise ValueError("component execution blocked by feasibility gate")
    if (
        not 0 <= index < 2
        or torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("one A100 and a registered worker index required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic workspace must be configured before Python")
    torch.set_num_threads(12)
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    folder = root / phase / f"workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    write_once(folder / "hardware.json", asdict(capture_gpu_environment(torch)))
    cfg = RunConfig.from_mapping(plan["configs"]["full_mechanisms"])
    inputs = read(root / "inputs.json")
    records = []
    parity = {}
    try:
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        with torch.inference_mode():
            if phase == "canary":
                current = inputs[str(protocol["consumed_sample_indices"][index])]
                other = inputs[str(protocol["consumed_sample_indices"][1 - index])]
                prepared = adapter.prepare_inputs(
                    build_multimodal_messages(
                        current["prompt"],
                        tuple(MediaItem(MediaKind.IMAGE, x["path"]) for x in current["media"]),
                    )
                )
                ordinary = compare_with_generate(adapter, prepared, max_new_tokens=16)
                identity = replace(
                    cfg, cache=replace(cfg.cache, retention_ratio=1.0, budget_value=2**31 - 1)
                )
                packed = compare_runtime_retention_one(adapter, prepared, identity)
                parity = dict(
                    ordinary=asdict(ordinary),
                    packed_retention_one=asdict(packed),
                    passed=all(
                        p.token_agreement == 1.0
                        and p.maximum_logit_difference <= protocol["retention_one_atol"]
                        for p in (ordinary, packed)
                    ),
                )
                write_once(folder / "parity.json", parity)
                if not parity["passed"]:
                    raise ValueError("registered FullKV/retention-one parity failed")
                order = [
                    ("full_kv", current),
                    ("retention_one", current),
                    ("full_mechanisms", current),
                    ("controlled_repair", current),
                    ("full_mechanisms", other),
                    ("full_mechanisms", current),
                ]
            else:
                assigned = protocol["component_sample_indices"][index::2]
                order = [
                    (arm, inputs[str(sample_index)])
                    for sample_index in assigned
                    for arm in protocol["component_arms"]
                ]
            for sequence, (arm, sample) in enumerate(order):
                arm_cfg = RunConfig.from_mapping(plan["configs"][arm])
                record = execute_arm(adapter, arm_cfg, sample, folder, arm, sequence)
                records.append(record)
                if phase == "canary" and arm == "retention_one":
                    ordinary_row = records[0]
                    runtime_identity = (
                        ordinary_row["tokens"] == record.get("tokens")
                        and ordinary_row["operations"]["prefill_logits_sha256"]
                        == record.get("operations", {}).get("prefill_logits_sha256")
                        and ordinary_row["operations"]["decode_logits_sha256"]
                        == record.get("operations", {}).get("decode_logits_sha256")
                    )
                    if not runtime_identity:
                        raise ValueError("independent-prefill runtime retention-one parity failed")
                # Fail fast before the expensive A-B-A continuation or expansion.
                if record["status"] != "completed" or (
                    record.get("checks") and not all_pass(record["checks"])
                ):
                    raise ValueError(f"execution/evidence gate failed for {arm}")
                gc.collect()
                torch.cuda.empty_cache()
        isolation = True
        if phase == "canary":
            a, repeated = records[2], records[5]
            isolation = (
                a["tokens"] == repeated["tokens"]
                and a["operations"]["decode_logits_sha256"]
                == repeated["operations"]["decode_logits_sha256"]
                and a["operations"]["prefill_logits_sha256"]
                == repeated["operations"]["prefill_logits_sha256"]
                and a["storage"] == repeated["storage"]
            )
        result = dict(
            status="completed",
            phase=phase,
            parity=parity,
            cache_isolation_aba_passed=isolation,
            record_count=len(records),
            all_checks_passed=isolation,
            raw_records=[r["run_id"] for r in records],
        )
    except Exception as error:
        result = dict(
            status="failed",
            phase=phase,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            record_count=len(records),
            all_checks_passed=False,
            parity=parity,
        )
    result.update(source_sha=plan["source_sha"], plan_sha256=expected)
    write_once(folder / "result.json", result)
    return 0 if result["all_checks_passed"] else 1


def analyze(root, expected, phase):
    plan = checked(root, expected)
    folder = root / f"{phase}_analysis"
    folder.mkdir(exist_ok=False)
    results, errors, rows = [], [], []
    for index in range(2):
        source = root / phase / f"workers/{index}/result.json"
        if not source.exists():
            errors.append(f"missing worker result: {index}")
            continue
        result = read(source)
        results.append(result)
        if not result.get("all_checks_passed"):
            errors.append(f"worker {index}: {result.get('error', 'isolation/control failed')}")
        for path in sorted(source.parent.glob("[0-9][0-9]-*.json")):
            if not path.name.endswith(".manifest.json"):
                row = read(path)
                if row["status"] != "completed":
                    errors.append(f"failed row: {row['run_id']}")
                if row.get("checks") and not all_pass(row["checks"]):
                    errors.append(f"mechanism gates: {row['run_id']} {row['checks']}")
                rows.append(row)
    expected_rows = 12 if phase == "canary" else 4 * len(plan["protocol"]["component_arms"])
    if len(rows) != expected_rows:
        errors.append(f"incomplete: {len(rows)}/{expected_rows} rows")
    decision = dict(
        phase=phase,
        all_gates_passed=not errors,
        errors=errors,
        row_count=len(rows),
        expected_rows=expected_rows,
        worker_results=results,
        classification=plan["classification"],
        paper_superiority_established=False,
        larger_campaign_authorized=False,
        source_sha=plan["source_sha"],
    )
    write_once(folder / "rows.json", rows)
    lines = [
        f"# Six-mechanism {phase} diagnostic",
        "",
        f"Passed: {not errors}",
        f"Rows: {len(rows)}/{expected_rows}",
        "",
        "Consumed inputs, instrumented runtime; not confirmatory quality/latency evidence.",
        "",
    ]
    for error in errors:
        lines.append(f"- {error}")
    if phase == "component" and not errors:
        # References are read only after every prediction is frozen.
        from mosaickv.evaluation.coco_shared_bytes import TokenizedCider
        from mosaickv.evaluation.metrics.types import MetricSample

        parent_path = Path(plan["protocol"]["parent_ledger"])
        if sha(parent_path) != plan["protocol"]["parent_ledger_sha256"]:
            raise ValueError("reference ledger changed before derived scoring")
        references = {
            sample["sample_id"]: tuple(sample["references"])
            for sample in read(parent_path)["samples"]
        }
        groups = {
            arm: [
                MetricSample(row["sample_id"], row["answer"], references[row["sample_id"]])
                for row in rows
                if row["arm"] == arm
            ]
            for arm in plan["protocol"]["component_arms"]
        }
        scorer = TokenizedCider([sample for samples in groups.values() for sample in samples])
        scores = {arm: scorer(samples) for arm, samples in groups.items()}
        write_once(
            folder / "descriptive_cider.json",
            dict(
                scores=scores,
                scored_per_arm=4,
                scorer="pycocoevalcap==1.2 PTB/CIDEr x100",
                reference_artifact_sha256=sha(parent_path),
                confirmation=False,
                no_statistical_superiority_claim=True,
            ),
        )
        lines += [
            "",
            "| Arm | CIDEr (4-image diagnostic) | Token agreement vs FullKV | "
            "Initial payload MiB | "
            "Final backing MiB | Repairs |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm in plan["protocol"]["component_arms"]:
            selected = [row for row in rows if row["arm"] == arm]
            reference = {row["sample_id"]: row for row in rows if row["arm"] == "full_kv"}
            agreement = sum(
                sum(
                    a == b
                    for a, b in zip(
                        row["tokens"], reference[row["sample_id"]]["tokens"], strict=True
                    )
                )
                / len(row["tokens"])
                for row in selected
            ) / len(selected)
            initial = (
                sum(row["storage"]["initial_active_kv_bytes"] for row in selected)
                / len(selected)
                / 2**20
            )
            final = (
                sum(row["storage"]["final_active_kv_storage_bytes"] for row in selected)
                / len(selected)
                / 2**20
            )
            repairs = sum(row["metrics"]["repair_count"] or 0 for row in selected)
            lines.append(
                f"| {arm} | {scores[arm]:.3f} | {agreement:.4f} | "
                f"{initial:.3f} | {final:.3f} | {repairs} |"
            )
        lines += [
            "",
            "Token agreement is not an official task score or proof of quality preservation.",
            "Graph-coverage-off ablates only the coverage reward, not graph construction.",
        ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    write_once(folder / "decision.json", decision)
    if phase == "canary" and not errors:
        launch(root, "component")
    print(json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--phase", default="canary", choices=("canary", "component"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "worker":
        return worker(args.root, args.plan_sha, args.phase, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.action == "analyze":
        analyze(args.root, args.plan_sha, args.phase)
    else:
        checked(args.root, args.plan_sha)
        return subprocess.call(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / f"{args.phase}_admission.json"),
                "--state",
                str(args.root / f"{args.phase}_admission_state.json"),
                "--apply",
                "--watch",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
