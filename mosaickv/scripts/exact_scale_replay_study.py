#!/usr/bin/env python3
"""Finite corrected-predicate + exact-header gate on two consumed inputs only."""

import argparse
import gzip
import json
import os
import subprocess
import traceback
from pathlib import Path

import header_predicate_study as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, PYTHON = common.SOURCE, common.PYTHON
PARENT = Path("/scratch/djy8hg/runs/evicastkv/exact-header-predicate-20261108-v1")
common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/exact_scale_replay.sbatch"
ARMS = ["complete_predicate", "complete_predicate_headers"]
EXTRA_TESTS = ()


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable root required")
    from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

    p = common.read(PARENT / "plan.json")
    controls = []
    for i in (0, 1):
        folder = PARENT / f"model/{i}"
        rows = [json.loads(x) for x in (folder / "raw.jsonl").read_text().splitlines()]
        row = next(r for r in rows if r["arm"] == "control_common")
        if not (
            row["status"] == "completed"
            and row["numerically_admissible"]
            and row["whole_request_certificate_valid"]
            and row["headers_sha"] == sha(folder / "headers.pt")
            and row["ledger_sha"] == sha(Path(row["ledger_path"]))
        ):
            raise ValueError("complete numeric/ownership/header control required")
        with gzip.open(row["ledger_path"], "rt") as stream:
            if not valid_whole_request_certificate(json.load(stream)):
                raise ValueError("parent ownership certificate failed independent check")
        paths = [folder / "raw.jsonl", folder / "headers.pt", Path(row["ledger_path"])]
        p["bound_files"].update({str(path): sha(path) for path in paths})
        controls.append(row)
    p["bound_files"][str(PARENT / "plan.json")] = sha(PARENT / "plan.json")
    p.update(
        source_sha=clean_source(SOURCE),
        study="complete_predicate_contract_repair_and_exact_scale_replay_v1",
        arms=ARMS,
        measurement_type="consumed_common_exact_header_replay_ablation",
        predecessor_fusion_failed="explicit chunk hint rejected; unchanged full-span policy",
        predecessor_valid_controls_only_reused=True,
        original_control_rows=controls,
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
        header_encoder="CPU exact-word dictionary with online transfer cost included",
        scale_lossless=True,
        CPU_residual_bytes=0,
    )
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = common.submit(root, "cpu")
    write_once(
        root / "submission.json", dict(cpu=cpu, follow=common.submit(root, "after_cpu", cpu))
    )
    print(json.dumps(common.read(root / "submission.json")), flush=True)


def header_feasibility(p):
    import torch

    from mosaickv.evaluation.exact_scale_codec import encode_cpu

    cases = []
    for i in (0, 1):
        path = PARENT / f"model/{i}/headers.pt"
        if sha(path) != p["bound_files"][str(path)]:
            raise ValueError("captured original headers changed")
        rows = torch.load(path, weights_only=True, map_location="cpu")
        old = new = 0
        roles, kinds = {}, {}
        for (_, role), body in rows.items():
            x = body["scale"]
            packet = encode_cpu(x)
            old += x.untyped_storage().nbytes()
            new += packet.storage_bytes
            if packet.kind != "raw":
                roles[role] = max(roles.get(role, 0), x.numel() * x.element_size())
            kinds[packet.kind] = kinds.get(packet.kind, 0) + 1
            if body["minimum"].dtype != torch.bfloat16:
                raise ValueError("do not double count already compressed BF16 minima")
        scratch = sum(roles.values())
        cases.append(
            dict(
                input=i,
                original_scale_bytes=old,
                encoded_scale_bytes=new,
                charged_replay_bytes=scratch,
                net_saving_bytes=old - new - scratch,
                kinds=kinds,
            )
        )
    return dict(
        measurement_type="derived_CPU_feasibility_not_model_memory",
        cases=cases,
        both_inputs_positive=all(r["net_saving_bytes"] > 0 for r in cases),
        online_encoding_not_yet_timed=True,
    )


def cpu(root, p):
    from packed_history_study import TESTS

    names = [
        *TESTS,
        "test_exact_scale_codec.py",
        "test_complete_transport_predicate.py",
        "test_header_predicate_study.py",
        "test_layer_code_delta.py",
        "test_exact_scale_replay.py",
        "test_exact_scale_replay_study.py",
        *EXTRA_TESTS,
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
                *["mosaickv/tests/unit/" + name for name in names],
            ],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    feasibility = header_feasibility(p)
    write_once(root / "header-feasibility.json", feasibility)
    passed = not done.returncode and feasibility["both_inputs_positive"]
    write_once(
        root / "cpu.json",
        dict(
            status="completed" if passed else "failed",
            returncode=done.returncode,
            feasibility_positive=feasibility["both_inputs_positive"],
        ),
    )


def operator(root, p, folder, i):
    import torch
    from compact_hybrid_study import manifest
    from packed_history_study import trials

    from mosaickv.evaluation.complete_transport_predicate import gpu_controls
    from mosaickv.evaluation.exact_scale_codec import encode_cpu, unpack_into

    manifest(p, sha(root / "plan.json"), folder, i, gpu=True)
    controls = gpu_controls()
    rows = torch.load(PARENT / f"model/{i}/headers.pt", weights_only=True, map_location="cpu")
    workspace = torch.empty(
        max(b["scale"].numel() for b in rows.values()) + 17, dtype=torch.float32, device="cuda"
    )
    records = []
    for (layer, role), body in rows.items():
        packet = encode_cpu(body["scale"]).to("cuda")
        workspace.fill_(-91)
        unpack_into(packet, workspace)
        actual = workspace[: body["scale"].numel()].view(body["scale"].shape).cpu()
        if not torch.equal(actual.view(torch.int32), body["scale"].view(torch.int32)):
            raise RuntimeError("actual model scale header changed an FP32 bit")
        if not bool((workspace[body["scale"].numel() :] == -91).all()):
            raise RuntimeError("actual header unpack exceeded the owned scratch")
        records.append(
            dict(
                layer=layer,
                role=role,
                kind=packet.kind,
                exact_FP32_bits=True,
                all_replay_sentinels_intact=True,
                trials=trials(lambda packet=packet: unpack_into(packet, workspace), p),
            )
        )
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            passed=True,
            model_weights_loaded=False,
            predicate_controls=controls,
            actual_header_controls=records,
        ),
    )


def analyze(root, p):
    from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

    rows, errors = [], []
    keys = [
        "initial_GPU_KV_bytes",
        "final_GPU_KV_bytes",
        "whole_request_KV_plus_workspace_peak_bytes",
    ]
    pairs = []
    for i in (0, 1):
        path = root / f"model/{i}/raw.jsonl"
        local = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if [r["arm"] for r in local] != ARMS:
            errors.append(dict(input=i, error="missing/duplicate registered rows"))
        for r in local:
            if not (r["status"] == "completed" and r.get("numerically_admissible")):
                errors.append(dict(input=i, arm=r["arm"], error=r.get("error", "numeric failure")))
            else:
                path = Path(r["ledger_path"])
                if sha(path) != r["ledger_sha"]:
                    errors.append(dict(input=i, error="ownership ledger hash changed"))
                with gzip.open(path, "rt") as stream:
                    if not valid_whole_request_certificate(json.load(stream)):
                        errors.append(dict(input=i, error="ownership certification failed"))
            rows.append(dict(input=i, **{k: v for k, v in r.items() if k != "detail"}))
        if len(local) == 2 and all(r["status"] == "completed" for r in local):
            control, candidate = local
            accepted = candidate.get("common_header")
            if isinstance(accepted, dict):
                early = accepted.get("early", {})
                if not early.get("packed_layers") or not early.get("downstream_headers_restored"):
                    errors.append(dict(input=i, error="early header lifetime path not complete"))
                accepted = accepted.get("late")
            engaged = bool(accepted and len(accepted) == 1 and accepted[0].get("enabled"))
            if not engaged:
                errors.append(dict(input=i, error="exact header codec did not engage"))
            reductions = {
                k: 1 - candidate[k] / control[k]
                if isinstance(candidate.get(k), (int, float))
                and isinstance(control.get(k), (int, float))
                and control[k] > 0
                else None
                for k in keys
            }
            pairs.append(dict(input=i, reductions=reductions, header_engaged=engaged))
    complete = len(rows) == 4 and not errors
    saving = (
        complete
        and len(pairs) == 2
        and all(all(x is not None and x > 0 for x in pair["reductions"].values()) for pair in pairs)
    )
    folder = root / "analysis"
    folder.mkdir(exist_ok=False)
    write_once(
        folder / "decision.json",
        dict(
            complete=complete,
            all_initial_final_peak_savings=saving,
            errors=errors,
            rows=rows,
            pairs=pairs,
            independent_quality_established=False,
            strict_latency_established=False,
            six_component_superiority_established=False,
            micro_prototype_or_history_followup_authorized=saving,
        ),
    )
    lines = [
        "# Common exact headers and corrected predicate",
        "",
        "Two consumed inputs only. No independent quality or clean latency claim.",
        "",
        f"Complete correctness/ownership: {complete}; all storage boundaries smaller: {saving}.",
        "",
        "| Input | Arm | Initial MiB | Final MiB | KV/workspace peak MiB |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        values = [f"{r[k] / 2**20:.6f}" if r.get(k) else "uncertified" for k in keys]
        lines.append(f"| {r['input']} | {r['arm']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Peak includes requested PyTorch KV/workspace; allocator rounding, vendor "
        "workspace and process HBM are not certified. Online CPU encoding, transfers, "
        "palettes/selectors and shared scale buffers are charged. Failures are retained. "
        "This common optimization must also apply to compatible baselines. "
        "Instrumented single-request timings do not establish latency superiority.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def model(root, p, folder, i):
    from mosaickv.evaluation.exact_scale_replay import use

    def context(stack, arm):
        return stack.enter_context(use()) if arm == ARMS[1] else None

    common.extra_model_context = context
    common.model(root, p, folder, i)


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
    p = common.checked(root, args.plan_sha)
    if args.action == "cpu":
        cpu(root, p)
    elif args.action == "after_cpu":
        if (root / "cpu.json").exists() and common.read(root / "cpu.json")["status"] == "completed":
            common.launch(root, "operator")
        else:
            write_once(root / "CPU-negative.json", dict(no_GPU_submission=True))
    elif args.action == "after_operator":
        results = [
            common.read(root / f"operator/{i}/result.json")
            if (root / f"operator/{i}/result.json").exists()
            else {}
            for i in (0, 1)
        ]
        passed = all(r.get("passed") is True for r in results)
        write_once(root / "operator-decision.json", dict(passed=passed, results=results))
        if passed:
            common.launch(root, "model")
    elif args.action in ("operator", "model"):
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(p["seed"])
        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        folder = root / f"{args.action}/{i}"
        folder.mkdir(parents=True, exist_ok=False)
        try:
            (operator if args.action == "operator" else model)(root, p, folder, i)
        except Exception:
            write_once(folder / "result.json", dict(status="failed", error=traceback.format_exc()))
            raise
    elif args.action == "analyze":
        analyze(root, p)
    else:
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
            cwd=SOURCE,
        )


if __name__ == "__main__":
    main()
