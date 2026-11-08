#!/usr/bin/env python3
"""Finite CPU-only source-dictionary screen on two consumed code captures."""

import argparse
import importlib.metadata
import json
import subprocess
import traceback
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/all-head-source-feasibility-20261107-v1")
SHELL = SOURCE / "mosaickv/slurm/source_dictionary_study.sbatch"


def read(path):
    return json.loads(path.read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable dictionary protocol changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if any(sha(Path(k)) != v for k, v in plan["bound_files"].items()):
        raise ValueError("consumed original source capture changed")
    return plan


def submit(root, action, after=None):
    cmd = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--partition=bii",
        "--cpus-per-task=4",
        "--mem=8G",
        "--time=00:20:00",
        f"--job-name=mk-source-dictionary-{action}",
    ]
    if action == "worker":
        cmd += ["--array=0-1%2"]
    if after:
        cmd += [f"--dependency=afterany:{after}"]
    cmd += [
        f"--output={root}/logs/{action}-%A_%a.out",
        f"--error={root}/logs/{action}-%A_%a.err",
        str(SHELL),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    return subprocess.check_output(cmd, text=True).strip()


def prepare(root, facility=False):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable CPU screen root required")
    paths = [PARENT / "plan.json", PARENT / "analysis/decision.json"]
    for i in (0, 1):
        paths += [
            PARENT / f"workers/{i}/{n}"
            for n in (
                "source_codes.pt",
                "source_codes.json",
                "manifest.json",
            )
        ]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu.txt").open("x") as out:
        test = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_source_dictionary.py",
                "mosaickv/tests/unit/test_all_head_source_codec.py",
            ],
            cwd=SOURCE,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if test.returncode:
        raise ValueError("CPU regression failure; no Slurm submission")
    paths.append(root / "cpu.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(path): sha(path) for path in paths},
            parent=str(PARENT),
            samples=2,
            layers_per_input=5 if facility else 32,
            layers=[0, 8, 16, 24, 31] if facility else list(range(32)),
            dictionary_anchor_count=4,
            anchor_selection="gain_facility" if facility else "uniform_original_source_rows",
            bounded_facility_candidates=16,
            row_symbols=4096,
            unchanged_code_bits=3,
            comparison="actual_best_of_legacy_byte_and_all_head_symbol_codec_with_selector",
            minimum_source_code_savings_fraction_for_operator_work=0.02,
            complete_dictionary_and_index_storage_charged=True,
            benchmark_labels_accessed=False,
            seed=20261107,
            no_fresh_or_paper_campaign_authorized=True,
            native_GPU_followup_automatically_submitted=False,
            measurement_type="CPU_all_consumer_exact_source_code_dictionary_screen",
        ),
    )
    array = submit(root, "worker")
    receipt = dict(array=array, analyzer=submit(root, "analyze", array), GPUs=0)
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def worker(root, expected, index):
    import numpy as np
    import torch
    from evicastkv_campaign import environment

    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.evaluation.source_dictionary import (
        DictionaryCodes,
        common_encode,
        decode,
        encode,
        facility_anchors,
    )

    p = checked(root, expected)
    folder = root / f"workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    origin = Path(p["parent"]) / f"workers/{index}"
    manifest = read(origin / "manifest.json")
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            source_dirty=False,
            config_sha=expected,
            origin_model_manifest=manifest,
            model_id="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            dataset=manifest["dataset"],
            dataset_revision=manifest["dataset"]["revision"],
            source_capture_scope="consumed_PrefixKV_Description_and_MM_Vet_not_COCO",
            environment=environment(),
            numpy_version=importlib.metadata.version("numpy"),
            CUDA_build=torch.version.cuda,
            CUDA_runtime="not_used_CPU_only",
            driver="not_used",
            gpu_type="not_used_CPU_only",
            gpu_count=0,
            backend="CPU_source_code_fixture",
            attention_implementation="not_used",
            seed=p["seed"],
            measurement_type=p["measurement_type"],
            vllm="not_used",
            sglang="not_used",
            **scheduler_identity(manifest),
        ),
    )
    rows, errors, packets = [], [], {}
    try:
        archive = torch.load(origin / "source_codes.pt", map_location="cpu", weights_only=True)
        if sorted(archive) != list(range(32)):
            raise ValueError("all original source layers required")
        for layer in p["layers"]:
            start = perf_counter()
            current = archive[layer].numpy()
            previous = None if layer == 0 else archive[layer - 1].numpy()
            anchors, facility = None, None
            if p["anchor_selection"] == "gain_facility":
                anchors, facility = facility_anchors(
                    previous,
                    current,
                    row_symbols=p["row_symbols"],
                    maximum=p["dictionary_anchor_count"],
                    candidates=p["bounded_facility_candidates"],
                )
            if anchors == ():
                packet = common_encode(previous, current)
                proof = dict(
                    dictionary_accepted=False,
                    common_source_code_bytes=packet.serialized_bytes + 1,
                    dictionary_source_code_bytes=packet.serialized_bytes + 1,
                    dictionary_predicted_groups=0,
                    dictionary_proposal_materialized=False,
                    benchmark_labels_accessed=False,
                )
            else:
                packet, proof = encode(
                    previous,
                    current,
                    row_symbols=p["row_symbols"],
                    anchor_count=p["dictionary_anchor_count"],
                    anchor_rows=anchors,
                )
            proof["facility_anchor_evidence"] = facility
            if not np.array_equal(decode(previous, packet), current):
                raise ValueError(f"source code changed at layer {layer}")
            if not np.array_equal(decode(previous, common_encode(previous, current)), current):
                raise ValueError("common-codec control changed source bits")
            arrays = (
                ("offsets", "payload", "anchors", "row_anchor")
                if isinstance(packet, DictionaryCodes)
                else ("offsets", "payload")
            )
            packets[layer] = dict(
                kind=type(packet).__name__,
                length=packet.length,
                raw=False if isinstance(packet, DictionaryCodes) else packet.raw,
                row_symbols=p["row_symbols"],
                selector=torch.tensor(
                    [
                        2
                        if isinstance(packet, DictionaryCodes)
                        else int(type(packet).__name__ == "SymbolCodes")
                    ],
                    dtype=torch.uint8,
                ),
                **{k: torch.from_numpy(getattr(packet, k).copy()) for k in arrays},
            )
            rows.append(
                dict(
                    layer=layer,
                    **proof,
                    selected_source_code_bytes=packet.serialized_bytes + 1,
                    original_code_bits_exact=True,
                    CPU_encode_and_controls_seconds=perf_counter() - start,
                )
            )
        torch.save(packets, folder / "packets.pt")
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed",
            errors=errors,
            layers=rows,
            packet_archive_sha=sha(folder / "packets.pt")
            if (folder / "packets.pt").exists()
            else None,
            native_GPU_or_quality_claim=False,
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    p = checked(root, expected)
    errors, results = [], {}
    for i in (0, 1):
        try:
            result = read(root / f"workers/{i}/result.json")
            if result["status"] != "completed" or len(result["layers"]) != p["layers_per_input"]:
                raise ValueError(result)
            if sha(root / f"workers/{i}/packets.pt") != result["packet_archive_sha"]:
                raise ValueError("serialized CPU packet archive changed")
            common = sum(r["common_source_code_bytes"] for r in result["layers"])
            selected = sum(r["selected_source_code_bytes"] for r in result["layers"])
            attempted = sum(r["dictionary_source_code_bytes"] for r in result["layers"])
            results[str(i)] = dict(
                common_source_code_bytes=common,
                selected_source_code_bytes=selected,
                all_dictionary_source_code_bytes=attempted,
                saved_bytes=common - selected,
                savings_fraction=(common - selected) / common,
                accepted_layers=sum(r["dictionary_accepted"] for r in result["layers"]),
                original_code_bits_exact=all(
                    r["original_code_bits_exact"] for r in result["layers"]
                ),
                CPU_total_encode_and_controls_seconds=sum(
                    r["CPU_encode_and_controls_seconds"] for r in result["layers"]
                ),
            )
        except Exception:
            errors.append(traceback.format_exc())
    passed = not errors and all(
        r["savings_fraction"] >= p["minimum_source_code_savings_fraction_for_operator_work"]
        and r["original_code_bits_exact"]
        for r in results.values()
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not errors,
            errors=errors,
            results=results,
            CPU_storage_gate_passed=passed,
            source_sha=p["source_sha"],
            common_storage_optimization_not_six_mechanism_quality_gain=True,
            GPU_KV_memory_latency_or_quality_superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
            action=(
                "complete_all_32_CPU_layers_before_GPU"
                if p["layers_per_input"] != 32
                else "bounded_native_operator_check_only"
            )
            if passed
            else "retain_common_codec_no_GPU_retry",
        ),
    )
    print(json.dumps(dict(complete=not errors, CPU_storage_gate_passed=passed, results=results)))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action", choices=("prepare", "worker", "analyze"))
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--plan-sha")
    a.add_argument("--facility", action="store_true")
    args = a.parse_args()
    if args.action == "prepare":
        return prepare(args.root, facility=args.facility)
    if args.action == "analyze":
        return analyze(args.root, args.plan_sha)
    import os

    return worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))


if __name__ == "__main__":
    raise SystemExit(main())
