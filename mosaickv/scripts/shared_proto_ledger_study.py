#!/usr/bin/env python3
"""Finite two-consumed-input architecture/allocator audit, no fresh campaign."""

import argparse
import gc
import json
import os
import subprocess
import traceback
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/fused-six-hybrid-fill-20261108-v1")
TESTS = (
    "test_shared_head_prototypes.py",
    "test_shared_head_scope.py",
    "test_kv_timeline.py",
    "test_request_ownership.py",
    "test_request_peak_coverage.py",
    "test_scratch_audit_lifetime.py",
    "test_shared_source_map_accounting.py",
    "test_source_replacement.py",
    "test_layer_code_delta.py",
    "test_predictive_source.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def check(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable source-prototype plan changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    for name, value in p["bound_files"].items():
        if sha(Path(name)) != value:
            raise ValueError("consumed fixture/control changed: " + name)
    return p


def submit(root, action, *, after=None):
    gpu = action == "worker"
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--job-name=mk-shared-" + action,
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
            "--mem=16G",
            "--time=01:10:00" if action == "controller" else "--time=00:10:00",
        ]
    if after:
        args.append("--dependency=afterany:" + after)
    args += [
        f"--output={root}/logs/{action}-%A_%a.out",
        f"--error={root}/logs/{action}-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/shared_proto_ledger.sbatch"),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    return subprocess.check_output(args, text=True).strip()


def prepare(root, reuse=None):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new write-once diagnostic root required")
    p = read(PARENT / "plan.json")
    decision = read(PARENT / "model-analysis/decision.json")
    if not decision["complete"] or not decision["execution_gate_passed"]:
        raise ValueError("complete previous numerical controls required")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu-tests.txt").open("x") as stream:
        done = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                *["mosaickv/tests/unit/" + name for name in TESTS],
            ],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if done.returncode:
        raise ValueError("CPU regressions failed; no GPU submission")
    bound = dict(p["bound_files"])
    bound.update(
        {
            str(path): sha(path)
            for path in (
                PARENT / "plan.json",
                PARENT / "model-analysis/decision.json",
                root / "cpu-tests.txt",
                SOURCE / "mosaickv/docs/shared_prototype_ownership.md",
            )
        }
    )
    # Rules referencing upstream line numbers/internal allocator APIs are
    # meaningless without the exact installed source identities.
    import torch
    import transformers

    for path in (
        Path(torch.__file__).parent / "cuda/memory.py",
        Path(torch.__file__).parent / "_C/__init__.pyi",
        Path(transformers.__file__).parent / "models/llama/modeling_llama.py",
    ):
        bound[str(path)] = sha(path)
    p.update(
        source_sha=source,
        bound_files=bound,
        group_size=8,
        source_map_policy="first_layer_shared",
        maximum_allocator_events=500000,
        warmups=0,
        repetitions=1,
        clean_latency_benchmark_authorized=False,
        measurement_type="consumed_all_head_source_prototype_and_time_aligned_ownership_diagnostic",
        fresh_or_paper_campaign_authorized=False,
        benchmarks_not_used_to_select_proposals=True,
    )
    if reuse is not None:
        previous = read(reuse / "plan.json")
        if previous.get("source_map_policy") != "first_layer_shared":
            raise ValueError("same immutable source-map algorithm required for audit-only reuse")
        allowed = {
            "mosaickv/scripts/shared_proto_ledger_study.py",
            "mosaickv/src/mosaickv/evaluation/request_ownership.py",
            "mosaickv/src/mosaickv/measurements/kv_timeline.py",
            "mosaickv/tests/unit/test_kv_timeline.py",
            "mosaickv/tests/unit/test_request_ownership.py",
            "mosaickv/docs/shared_prototype_ownership.md",
        }
        changed = set(
            subprocess.check_output(
                ["git", "diff", "--name-only", previous["source_sha"], source],
                cwd=SOURCE,
                text=True,
            ).splitlines()
        )
        if not changed <= allowed:
            raise ValueError("audit-only retry cannot change numerical source policy")
        (root / "reused-canaries").mkdir()
        for i in (0, 1):
            folder = reuse / f"workers/{i}"
            canaries = [
                r
                for r in map(json.loads, (folder / "raw.jsonl").read_text().splitlines())
                if r["phase"] == "canary"
            ]
            if (
                len(canaries) != 2
                or any(
                    r["status"] != "completed" or not r["numerically_admissible"] for r in canaries
                )
                or not read(folder / "native.json")["passed"]
            ):
                raise ValueError("complete native and numerical canaries required")
            receipt = root / f"reused-canaries/{i}.json"
            write_once(
                receipt,
                dict(
                    producer_source_sha=previous["source_sha"],
                    parent=str(reuse),
                    rows=canaries,
                    native=read(folder / "native.json"),
                ),
            )
            for path in (
                receipt,
                folder / "manifest.json",
                folder / "native.json",
                folder / "storage_compact.logits.pt",
                folder / "shared_source_prototypes.logits.pt",
            ):
                p["bound_files"][str(path)] = sha(path)
        p.update(
            audit_only=True,
            reuse_canaries=str(root / "reused-canaries"),
            canary_inference_not_repeated=True,
        )
    write_once(root / "plan.json", p)
    array = submit(root, "worker")
    write_once(
        root / "admission.json",
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
    receipt = dict(
        array=array,
        controller=submit(root, "controller"),
        analyzer=submit(root, "analyze", after=array),
        maximum_A100s=2,
        CPUs_per_worker=12,
    )
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def native(folder, seed):
    import torch

    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.precision_tier import pack, unpack_cpu
    from mosaickv.evaluation.shared_head_prototypes import construct
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction, project
    from mosaickv.measurements.kv_timeline import CudaOwnershipSession

    torch.manual_seed(seed)
    n, h, d = 128, 32, 128
    x = torch.randn(1, 1, n, h * d).bfloat16()
    cpu = pack(x, bits=3, axis=2)
    from dataclasses import replace

    body = replace(cpu, codes=cpu.codes.cuda(), minimum=cpu.minimum.cuda(), scale=cpu.scale.cuda())
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    cos = torch.ones(1, n, d, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    exact_ids = torch.tensor([0], device="cuda", dtype=torch.int32)
    zero = torch.zeros(1, 1, 1, d, device="cuda", dtype=torch.bfloat16)
    packet = SharedSourceFormat(
        body,
        (1, h, n, d),
        exact_ids,
        (encode(zero), encode(zero)),
        cos,
        sin,
        torch.arange(n, dtype=torch.int32, device="cuda"),
        torch.arange(n, dtype=torch.int16, device="cuda")[None].expand(h, n),
    )
    original = tuple(
        torch.empty(packet.shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    # Compile before tracing; this bounded operator ledger is NOT full-request
    # certification and contains no unrelated concurrent model operations.
    project(packet, attention, original)
    del zero
    gc.collect()
    torch.cuda.synchronize()
    with CudaOwnershipSession(max_entries=20000) as audit:
        audit.mark(
            (*attention.k_proj.parameters(), *attention.v_proj.parameters()),
            "model",
            "frozen toy native projection weights",
        )
        audit.mark(
            (*packet.tensors(), *original),
            "kv_persistent",
            "original all-head source, headers, positions and projection control",
        )
        with audit.serialized_region(
            "kv_workspace", "allocation-isolated source prototype transaction and verification"
        ):
            transaction = construct(body, (tuple(range(8, 24)),), heads=h, mandatory=(0, n - 1))
            candidate = replace(packet, source=transaction.body)
            decoded = decode_source_for_construction(candidate)
            expected = unpack_cpu(
                replace(
                    transaction.body,
                    codes=transaction.body.codes.cpu(),
                    minimum=transaction.body.minimum.cpu(),
                    scale=transaction.body.scale.cpu(),
                    row_map=transaction.body.row_map.cpu(),
                )
            )[:, 0]
            source_parity = bool(
                torch.equal(decoded.cpu().view(torch.int16), expected.view(torch.int16))
            )
            outputs = tuple(torch.empty_like(original[0]) for _ in range(2))
            project(candidate, attention, outputs)
            projected_error = []
            for component, weight in enumerate((attention.k_proj, attention.v_proj)):
                reference = weight(decoded).view(1, n, h, d).transpose(1, 2)
                projected_error.append(
                    float((outputs[component].float() - reference.float()).abs().max())
                )
                torch.testing.assert_close(outputs[component], reference, atol=0.015625, rtol=0.01)
            # Drop every original code/packet owner, not just one head view.
            del body, packet
            gc.collect()
            retired = not transaction.record()["old_source_tensor_alive"]
            compressed_record = transaction.record()
            restored = transaction.promote()
            restored_packet = replace(candidate, source=restored)
            project(restored_packet, attention, outputs)
            repaired = all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(outputs, original, strict=True)
            )
            result = dict(
                source_parity=source_parity,
                old_source_tensor_retired=retired,
                all_64_projection_consumers_verified=True,
                exact_repair_projection_bits=repaired,
                maximum_projection_errors=projected_error,
                replacement=compressed_record,
                repaired_state=transaction.record(),
            )
    result["ownership_timeline"] = audit.report(
        opaque_workspace_classified=True, scope="bounded_operator"
    )
    result["passed"] = (
        source_parity
        and retired
        and repaired
        and result["ownership_timeline"]["exact_scoped_KV_peak_certified"]
    )
    write_once(folder / "native.json", result)
    return result


def model(p, folder, i):
    import evicastkv_paper_precision as campaign
    import torch
    from compact_hybrid_study import fidelity

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.paper_request_certification import WholeRequestEnvelope
    from mosaickv.evaluation.predictive_source_gpu import use as codec
    from mosaickv.evaluation.request_ownership import audit_request
    from mosaickv.evaluation.shared_head_scope import use as prototype
    from mosaickv.hf_cli import _model_kwargs

    controls = p["controls"]
    cfg = RunConfig.from_mapping(controls["parent"]["config"])
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    transport = dict(controls["parent"], options=controls["options"])
    features = read(controls["inputs"])[i]
    reference = torch.load(
        controls["references"][str(i)]["storage"], weights_only=True, map_location="cpu"
    )
    records = []
    with (folder / "raw.jsonl").open("x") as out:

        def run(arm, *, audit=False):
            row = dict(
                arm=arm,
                phase="ownership_audit" if audit else "canary",
                status="failed",
                sample_id=features["sample_id"],
                manifest_path=str(folder / "manifest.json"),
            )
            proof = {}
            ledger = None
            mechanisms = proof
            start = perf_counter()
            try:
                gc.collect()
                torch.cuda.synchronize()
                with (
                    codec(decoder="packed_words", encoding_policy="direct_additive_compact_rare"),
                    prototype(
                        group_size=p["group_size"],
                        map_policy=p.get("source_map_policy", "independent"),
                    )
                    if arm == "shared_source_prototypes"
                    else nullcontext(proof) as mechanisms,
                    WholeRequestEnvelope(adapter.model) if audit else nullcontext() as envelope,
                    audit_request(adapter, max_entries=p["maximum_allocator_events"])
                    if audit
                    else nullcontext() as ledger,
                ):
                    logits, detail = campaign.execute(
                        adapter, features, transport, "mosaickv_exact", audit=audit
                    )
                torch.cuda.synchronize()
                actual = logits.cpu()
                numeric = fidelity(actual, reference, strict=arm == "storage_compact", p=p)
                torch.save(actual, folder / f"{arm}{'-audit' if audit else ''}.logits.pt")
                row.update(
                    status="completed",
                    numerically_admissible=numeric["passed"],
                    fidelity=numeric,
                    detail=detail,
                    mechanisms=mechanisms,
                    request_wall_seconds=perf_counter() - start,
                )
                if audit:
                    row["ownership_timeline"] = ledger
                    row["allocator_envelope"] = envelope.certificate
            except Exception:
                row.update(
                    error=traceback.format_exc(),
                    partial_mechanisms=mechanisms,
                    partial_ownership_timeline=ledger,
                    request_wall_seconds=perf_counter() - start,
                )
            out.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            out.flush()
            records.append(row)
            return row

        if p.get("audit_only"):
            run("storage_compact", audit=True)
            run("shared_source_prototypes", audit=True)
            return dict(
                completed=True,
                raw_rows=len(records),
                reused_canaries=True,
                no_larger_campaign_authorized=True,
            )
        control = run("storage_compact")
        candidate = run("shared_source_prototypes")
        # Always inspect accounting, even when approximation is negative; do
        # not spend 20 repetitions timing a numerically rejected candidate.
        run("storage_compact", audit=True)
        if candidate["status"] == "completed":
            run("shared_source_prototypes", audit=True)
        admitted = all(
            r["status"] == "completed" and r.get("numerically_admissible")
            for r in (control, candidate)
        )
    return dict(
        completed=True,
        numerical_gate_passed=admitted,
        raw_rows=len(records),
        no_larger_campaign_authorized=True,
    )


def worker(root, expected):
    import torch
    from compact_hybrid_study import manifest

    p = check(root, expected)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(p["seed"])
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed")
    try:
        manifest(p, expected, folder, i, gpu=True)
        control = (
            read(Path(p["reuse_canaries"]) / f"{i}.json")["native"]
            if p.get("audit_only")
            else torch.inference_mode()(native)(folder, p["seed"] + i)
        )
        if not control["passed"]:
            raise ValueError(
                "bounded native source-replacement/ownership gate failed; no model run"
            )
        # Retire native controls before allocating real model weights.
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        result.update(status="completed", model=model(p, folder, i))
    except Exception:
        result["error"] = traceback.format_exc()
    write_once(folder / "result.json", result)
    if result["status"] != "completed":
        raise SystemExit(1)


def analyze(root, expected):
    p = check(root, expected)
    rows, failures = [], []
    for i in (0, 1):
        result = root / f"workers/{i}/result.json"
        if not result.exists() or read(result)["status"] != "completed":
            failures.append(
                dict(input=i, error=read(result) if result.exists() else "missing worker")
            )
        raw = root / f"workers/{i}/raw.jsonl"
        if raw.exists():
            rows.extend(dict(input=i, **json.loads(line)) for line in raw.read_text().splitlines())
    canaries = [r for r in rows if r["phase"] == "canary"]
    failures.extend(
        dict(
            input=r["input"],
            arm=r["arm"],
            phase=r["phase"],
            error=r.get("error", "numerical gate failed"),
        )
        for r in rows
        if r["status"] != "completed"
    )
    if len(rows) != 8:
        failures.append(dict(error="expected eight complete canary/audit rows", observed=len(rows)))
    admitted = len(canaries) == 4 and all(
        r["status"] == "completed" and r.get("numerically_admissible") for r in canaries
    )
    memory = []
    for i in (0, 1):
        selected = {r["arm"]: r for r in canaries if r["input"] == i and r["status"] == "completed"}
        if len(selected) == 2:
            base, candidate = selected["storage_compact"], selected["shared_source_prototypes"]
            memory.append(
                dict(
                    input=i,
                    ratios={
                        k: candidate["detail"][k] / base["detail"][k]
                        for k in (
                            "initial_encoded_bytes",
                            "final_encoded_bytes",
                            "reported_counter_peak_bytes",
                        )
                    },
                    peak_ratio_is_decode_counter_not_whole_request_certificate=True,
                )
            )
        if p.get("audit_only"):
            rows.extend(
                dict(
                    input=i,
                    reused_canary=True,
                    producer_source_sha=read(Path(p["reuse_canaries"]) / f"{i}.json")[
                        "producer_source_sha"
                    ],
                    **row,
                )
                for row in read(Path(p["reuse_canaries"]) / f"{i}.json")["rows"]
            )
    summary = dict(
        complete=not failures,
        failures=failures,
        numerical_gate_passed=admitted,
        memory_comparison=memory,
        all_reported_cache_boundaries_smaller=len(memory) == 2
        and all(all(value < 1 for value in record["ratios"].values()) for record in memory),
        rows=rows,
        independent_quality_established=False,
        fresh_or_paper_campaign_authorized=False,
        strict_latency_advantage_established=False,
    )
    (root / "analysis").mkdir(exist_ok=False)
    write_once(root / "analysis/decision.json", summary)
    text = [
        "# Shared-owner prototype and KV timeline diagnostic",
        "",
        "Two consumed inputs only; not paper-quality evidence.",
        "",
        f"Complete: {not failures}; model numerical gate: {admitted}.",
        "",
        "| Input | Arm | Phase | Status | Initial MiB | Final MiB | Logit gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r.get("detail", {})

        def size(k, d=d):
            return f"{d[k] / 2**20:.6f}" if k in d else "not measured"

        text.append(
            f"| {r['input']} | {r['arm']} | {r['phase']} | {r['status']} | "
            f"{size('initial_encoded_bytes')} | {size('final_encoded_bytes')} | "
            f"{r.get('numerically_admissible', False)} |"
        )
    text += [
        "",
        "Exact request certification requires complete classified allocation generations "
        "and explicit workspace attribution. Unknowns remain blockers; operator-only "
        "certification is not a whole-request certificate.",
    ]
    with (root / "analysis/report.md").open("x") as stream:
        stream.write("\n".join(text) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.reuse)
    elif args.action == "worker":
        worker(args.root, args.plan_sha)
    elif args.action == "analyze":
        analyze(args.root, args.plan_sha)
    else:
        check(args.root, args.plan_sha)
        subprocess.run(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / "admission.json"),
                "--state",
                str(args.root / "admission-state.json"),
                "--apply",
                "--watch",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
