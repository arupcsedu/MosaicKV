#!/usr/bin/env python3
"""Finite CPU -> GPU operators -> same-consumed model controls -> header costs.

No fresh identities, paper campaign, unsupported native rows, label-based
selection or default-policy promotion. Every producer has an afterany reviewer.
"""

import argparse
import gc
import gzip
import json
import os
import subprocess
import traceback
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/packed-prototype-history-20261108-v5")
WORKER_WRAPPER = SOURCE / "mosaickv/slurm/header_predicate.sbatch"


def extra_model_context(stack, arm):
    """Default is unchanged; versioned common-codec studies bind an explicit scope."""
    return None


def assess_fidelity(actual, reference, p, arm):
    from compact_hybrid_study import fidelity

    return fidelity(actual, reference, strict=True, p=p)


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    p = read(root / "plan.json")
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable plan changed")
    clean_source(SOURCE, p["source_sha"])
    for name, digest in p["bound_files"].items():
        if sha(Path(name)) != digest:
            raise ValueError("immutable dependency changed: " + name)
    return p


def submit(root, action, after=None, stage=None):
    gpu = action in ("operator", "model")
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--job-name=mk-header-" + action,
    ]
    if gpu:
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=01:00:00",
            "--array=0-1%2",
            "--hold",
        ]
    else:
        args += [
            "--partition=bii",
            "--cpus-per-task=2",
            "--mem=24G",
            "--time=" + ("01:10:00" if action == "controller" else "00:15:00"),
        ]
    if after:
        args += ["--dependency=afterany:" + after]
    args += [
        f"--output={root}/logs/{action}-{stage or 'none'}-%A_%a.out",
        f"--error={root}/logs/{action}-{stage or 'none'}-%A_%a.err",
        str(WORKER_WRAPPER),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
        stage or "none",
    ]
    return subprocess.check_output(args, text=True).strip()


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable root required")
    p = read(PARENT / "plan.json")
    gate = read(PARENT / "analysis/decision.json")
    if not gate["complete"] or not gate["all_registered_component_gates_passed"]:
        raise ValueError("complete predecessor correctness evidence required")
    p.update(
        source_sha=clean_source(SOURCE),
        study="exact_scale_headers_complete_predicate_v1",
        measurement_type="consumed_common_header_predicate_ablation",
        result_category="unified_comparison",
        result_origin="locally_measured",
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
        arms=["control_common", "complete_predicate"],
        benchmark_labels_accessed=False,
        model_warmups=0,
        model_repetitions=1,
        operator_warmups=5,
        operator_repetitions=20,
        model_history_graph=False,
    )
    p["bound_files"].update(
        {
            str(PARENT / "plan.json"): sha(PARENT / "plan.json"),
            str(PARENT / "analysis/decision.json"): sha(PARENT / "analysis/decision.json"),
        }
    )
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = submit(root, "cpu")
    write_once(root / "submission.json", dict(cpu=cpu, follow=submit(root, "after_cpu", cpu)))
    print(json.dumps(read(root / "submission.json")), flush=True)


def launch(root, stage):
    array = submit(root, stage)
    write_once(
        root / f"admission-{stage}.json",
        dict(
            timezone="America/New_York",
            window="all_times",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[array],
            adopt_existing_holds=[array],
        ),
    )
    follow = "after_operator" if stage == "operator" else "analyze"
    write_once(
        root / f"{stage}-submission.json",
        dict(
            array=array,
            controller=submit(root, "controller", stage=stage),
            analyzer=submit(root, follow, array),
            GPUs_per_worker=1,
            CPUs_per_worker=12,
            maximum_workers=2,
        ),
    )


def cpu(root):
    from packed_history_study import TESTS

    tests = [
        *TESTS,
        "test_exact_scale_codec.py",
        "test_complete_transport_predicate.py",
        "test_header_predicate_study.py",
    ]
    with (root / "cpu-tests.txt").open("x") as stream:
        done = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                *["mosaickv/tests/unit/" + name for name in tests],
            ],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    write_once(
        root / "cpu.json",
        dict(status="completed" if not done.returncode else "failed", returncode=done.returncode),
    )


def operators(root, p, folder, index):
    import torch
    from compact_hybrid_study import manifest
    from packed_history_study import trials

    from mosaickv.evaluation import complete_transport_predicate as complete
    from mosaickv.evaluation.exact_scale_codec import encode_cpu, unpack_into

    manifest(p, sha(root / "plan.json"), folder, index, gpu=True)
    controls = complete.gpu_controls()
    x = torch.randn(1, 32, 603, 128, device="cuda").bfloat16()
    y = x.clone()
    y.flatten()[::127] += 0.25
    timings = {
        name: trials(lambda fn=fn: fn(x, y), p)
        for name, fn in (("full_union", complete.oracle), ("fused_union", complete.fused))
    }
    scale_controls = []
    for count, kind in ((129, "raw"), (4096, "word"), (4096, "low16")):
        values = (
            torch.rand(count)
            if kind == "raw"
            else (torch.arange(count) % 7 + 1).float() / 7
            if kind == "word"
            else (torch.arange(count) + 200).float() / 7
        )
        packet = encode_cpu(values).to("cuda")
        if packet.kind != kind:
            raise RuntimeError("registered scale operator did not exercise " + kind)
        out = torch.full((count + 17,), -99.0, device="cuda")
        unpack_into(packet, out)
        if not torch.equal(
            out[:count].cpu().view(torch.int32), values.view(torch.int32)
        ) or not bool((out[count:] == -99).all()):
            raise RuntimeError("GPU scale words/sentinels changed")
        scale_controls.append(
            dict(
                kind=kind,
                count=count,
                FP32_words_bitwise=True,
                sentinels=True,
                complete_encoded_bytes=packet.storage_bytes,
            )
        )
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            passed=True,
            predicate_controls=controls,
            scale_controls=scale_controls,
            trials=timings,
            model_weights_loaded=False,
        ),
    )


def model(root, p, folder, index):
    import evicastkv_paper_precision as campaign
    import torch
    from compact_hybrid_study import manifest

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.complete_transport_predicate import use as complete
    from mosaickv.evaluation.predictive_source_gpu import use as codec
    from mosaickv.evaluation.request_ownership import audit_request
    from mosaickv.evaluation.scale_capture import use as capture
    from mosaickv.evaluation.tiled_softmax import use as tile
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

    manifest(p, sha(root / "plan.json"), folder, index, gpu=True)
    cfg = RunConfig.from_mapping(p["controls"]["parent"]["config"])
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    transport = dict(
        p["controls"]["parent"],
        options={
            method: dict(options, native_history_graph=False)
            for method, options in p["controls"]["options"].items()
        },
    )
    sample = read(p["controls"]["inputs"])[index]
    reference = torch.load(
        p["controls"]["references"][str(index)]["storage"], weights_only=True, map_location="cpu"
    )
    results = []
    with (folder / "raw.jsonl").open("x") as output:
        for arm in p["arms"]:
            row = dict(
                arm=arm,
                method="evicastkv_common_" + arm,
                sample_id=sample["sample_id"],
                status="failed",
                manifest_path=str(folder / "manifest.json"),
                result_category="unified_comparison",
                result_origin="locally_measured",
                no_six_component_or_independent_quality_claim=True,
            )
            ledger = None
            started = perf_counter()
            try:
                gc.collect()
                torch.cuda.synchronize()
                with ExitStack() as stack:
                    stack.enter_context(
                        codec(
                            decoder="packed_words", encoding_policy="direct_additive_compact_rare"
                        )
                    )
                    tile_report = stack.enter_context(tile(rows=p["tiled_softmax_rows"]))
                    predicate_report = (
                        stack.enter_context(complete()) if arm != "control_common" else {}
                    )
                    common_header_report = extra_model_context(stack, arm)
                    headers = stack.enter_context(capture()) if arm == "control_common" else None
                    ledger = stack.enter_context(
                        audit_request(adapter, max_entries=p["maximum_allocator_events"])
                    )
                    logits, detail = campaign.execute(
                        adapter, sample, transport, "mosaickv_exact", audit=True
                    )
                torch.cuda.synchronize()
                actual = logits.cpu()
                proof = assess_fidelity(actual, reference, p, arm)
                torch.save(actual, folder / f"{arm}.logits.pt")
                path = folder / f"{arm}.ledger.json.gz"
                with gzip.open(path, "xt", compresslevel=1) as stream:
                    json.dump(ledger, stream, allow_nan=False, separators=(",", ":"))
                if headers is not None:
                    if len(headers) != 32 or set(layer for layer, role in headers) != set(
                        range(32)
                    ):
                        raise RuntimeError("complete real 32-layer header capture required")
                    torch.save(headers, folder / "headers.pt")
                row.update(
                    status="completed",
                    fidelity=proof,
                    numerically_admissible=bool(proof["passed"]),
                    initial_GPU_KV_bytes=detail["initial_encoded_bytes"],
                    final_GPU_KV_bytes=detail["final_encoded_bytes"],
                    whole_request_KV_plus_workspace_peak_bytes=ledger["exact_KV_peak_bytes"],
                    whole_request_certificate_valid=valid_whole_request_certificate(ledger),
                    ownership_blockers=ledger["blocks_exact_certification"],
                    ledger_path=str(path),
                    ledger_sha=sha(path),
                    CPU_residual_bytes=0,
                    predicate=predicate_report,
                    common_header=common_header_report,
                    tiled_softmax=tile_report,
                    headers_sha=sha(folder / "headers.pt") if headers is not None else None,
                    timing=dict(
                        instrumented_request_seconds=perf_counter() - started,
                        decode_cuda_ms=detail["decode_cuda_ms"],
                        NOT_clean_latency_measurement=True,
                    ),
                    used_fullkv_fallback=detail.get("used_fullkv_fallback"),
                    detail=detail,
                )
                row["detail"].update(
                    tensor_result_auditing=False,
                    decode_owned_peak_bytes=None,
                    peak_measurement_admissible=False,
                    legacy_tensor_result_audit_superseded=True,
                )
            except Exception:
                row.update(
                    error=traceback.format_exc(), worker_wall_seconds=perf_counter() - started
                )
                if ledger and "ledger_path" not in row:
                    path = folder / f"{arm}.partial-ledger.json.gz"
                    with gzip.open(path, "xt", compresslevel=1) as stream:
                        json.dump(ledger, stream, allow_nan=False, separators=(",", ":"))
                    row.update(partial_ledger_path=str(path), partial_ledger_sha=sha(path))
            results.append(row)
            output.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            output.flush()
            if arm == "control_common" and not row.get("numerically_admissible"):
                break
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            rows=len(results),
            failures=sum(r["status"] != "completed" for r in results),
        ),
    )


def header_costs(root):
    import torch

    from mosaickv.evaluation.exact_scale_codec import encode_cpu

    cases = []
    for i in (0, 1):
        path = root / f"model/{i}/headers.pt"
        rows = torch.load(path, map_location="cpu", weights_only=True)
        old = encoded = scratch = 0
        kinds = {}
        layers = []
        for (layer, role), body in sorted(rows.items()):
            packet = encode_cpu(body["scale"])
            before = body["scale"].untyped_storage().nbytes()
            after = packet.storage_bytes
            old += before
            encoded += after
            if packet.kind != "raw":
                scratch = max(scratch, body["scale"].numel() * 4)
            kinds[packet.kind] = kinds.get(packet.kind, 0) + 1
            layers.append(
                dict(
                    layer=layer,
                    role=role,
                    shape=list(body["scale"].shape),
                    original_scale_bytes=before,
                    encoded_scale_bytes=after,
                    kind=packet.kind,
                    bits=packet.bits,
                    minimum_already_bf16=body["minimum"].dtype == torch.bfloat16,
                )
            )
        cases.append(
            dict(
                input=i,
                source_archive=str(path),
                source_sha=sha(path),
                original_scale_bytes=old,
                encoded_scale_bytes=encoded,
                charged_shared_FP32_replay_bytes=scratch,
                net_persistent_saving_bytes=old - encoded - scratch,
                kinds=kinds,
                layers=layers,
            )
        )
    write_once(
        root / "header-costs.json",
        dict(
            measurement_type="derived_CPU_exact_scale_feasibility",
            cases=cases,
            both_inputs_net_positive=all(c["net_persistent_saving_bytes"] > 0 for c in cases),
            whole_model_memory_NOT_measured=True,
            no_new_quality_or_latency_claim=True,
        ),
    )


def analyze(root, p):
    from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

    rows = []
    errors = []
    for i in (0, 1):
        folder = root / f"model/{i}"
        if not (folder / "raw.jsonl").exists():
            errors.append(dict(input=i, error="missing worker"))
            continue
        local = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
        if [r["arm"] for r in local] != p["arms"]:
            errors.append(dict(input=i, error="missing/duplicate registered arms"))
        for row in local:
            if (
                row["status"] != "completed"
                or not row.get("numerically_admissible")
                or not row.get("whole_request_certificate_valid")
            ):
                errors.append(
                    dict(
                        input=i, arm=row["arm"], error=row.get("error", "numeric/ownership failure")
                    )
                )
            if row.get("status") == "completed":
                try:
                    path = Path(row["ledger_path"])
                    if sha(path) != row["ledger_sha"]:
                        raise ValueError("produced ledger changed")
                    with gzip.open(path, "rt") as stream:
                        if not valid_whole_request_certificate(json.load(stream)):
                            raise ValueError("independent ownership verification failed")
                    if (
                        row["arm"] == "control_common"
                        and sha(folder / "headers.pt") != row["headers_sha"]
                    ):
                        raise ValueError("actual header capture changed")
                except Exception as exc:
                    errors.append(dict(input=i, arm=row["arm"], error=str(exc)))
            rows.append(dict(input=i, **row))
        if len(local) == 2 and local[0]["status"] == local[1]["status"] == "completed":
            for key in ("initial_GPU_KV_bytes", "final_GPU_KV_bytes"):
                if local[0][key] != local[1][key]:
                    errors.append(
                        dict(input=i, error="predicate changed persistent storage: " + key)
                    )
    folder = root / "analysis"
    folder.mkdir(exist_ok=False)
    if not errors and len(rows) == 4:
        try:
            header_costs(root)
        except Exception:
            errors.append(dict(error="header cost analysis failed", detail=traceback.format_exc()))
    write_once(
        folder / "decision.json",
        dict(
            complete=len(rows) == 4 and not errors,
            errors=errors,
            rows=[{k: v for k, v in r.items() if k != "detail"} for r in rows],
            independent_quality_established=False,
            strict_latency_established=False,
            six_mechanism_superiority_established=False,
        ),
    )
    lines = [
        "# Common exact-header / complete-predicate diagnostic",
        "",
        "Two consumed COCO inputs; no independent benchmark-quality conclusion.",
        "",
        f"Complete and numerically valid: {len(rows) == 4 and not errors}.",
        "",
        "| Input | Arm | Initial MiB | Final MiB | Whole-request KV/workspace peak MiB | Guard |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        sizes = [
            f"{r[k] / 2**20:.6f}" if r.get(k) is not None else "uncertified"
            for k in (
                "initial_GPU_KV_bytes",
                "final_GPU_KV_bytes",
                "whole_request_KV_plus_workspace_peak_bytes",
            )
        ]
        lines.append(
            f"| {r['input']} | {r['arm']} | "
            + " | ".join(sizes)
            + f" | {r.get('numerically_admissible', False)} |"
        )
    lines += [
        "",
        "Peak is requested PyTorch-owned KV plus declared workspace, not allocator "
        "rounding/reserved/vendor/process HBM. Instrumented request times are not clean "
        "latency evidence. All failures retained. Header costs include actual "
        "index/palette/selector tensors and shared FP32 replay; online encoding and "
        "whole-model header effects still require execution. "
        "Independent validation remains blocked.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "after_cpu",
            "operator",
            "after_operator",
            "model",
            "analyze",
            "controller",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--stage")
    args = parser.parse_args()
    root = args.root
    if args.action == "prepare":
        prepare(root)
        return
    p = checked(root, args.plan_sha)
    if args.action == "cpu":
        cpu(root)
    elif args.action == "after_cpu":
        if (root / "cpu.json").exists() and read(root / "cpu.json")["status"] == "completed":
            launch(root, "operator")
        else:
            write_once(root / "CPU-negative.json", dict(no_GPU_submission=True))
    elif args.action == "after_operator":
        results = [
            read(root / f"operator/{i}/result.json")
            if (root / f"operator/{i}/result.json").exists()
            else {}
            for i in (0, 1)
        ]
        passed = all(r.get("passed") is True for r in results)
        write_once(root / "operator-decision.json", dict(passed=passed, results=results))
        if passed:
            launch(root, "model")
    elif args.action in ("operator", "model"):
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(p["seed"])
        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        folder = root / f"{args.action}/{i}"
        folder.mkdir(parents=True, exist_ok=False)
        try:
            (operators if args.action == "operator" else model)(root, p, folder, i)
        except Exception:
            write_once(folder / "result.json", dict(status="failed", error=traceback.format_exc()))
            raise
    elif args.action == "controller":
        subprocess.run(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(root / f"admission-{args.stage}.json"),
                "--state",
                str(root / f"admission-{args.stage}-state.json"),
                "--apply",
                "--watch",
            ],
            check=True,
        )
    else:
        analyze(root, p)


if __name__ == "__main__":
    main()
