#!/usr/bin/env python3
"""Finite immutable-pointer history batching: CPU -> canary -> paired controls."""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import evicastkv_consolidated_history as previous

from mosaickv.evaluation.fox_campaign import sha, write_once

runtime, base = previous.runtime, previous.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/segmented-history-consumed-20261030-v2")


def prepare(root, phase):
    original = runtime.encoded.history.original_writer

    def register(path, value):
        if path.name == "plan.json":
            value.update(
                protocol="immutable_pointer_segmented_exact_source_history_v1",
                source_history_comparison_kind="segmented",
                history_source_packets_copied_or_refit=False,
                source_packet_copy_overlap_charged=False,
                source_packet_merge=False,
                original_bit_checks_per_consolidated_layer=0,
                original_packet_three_bit_checks_unchanged=True,
                same_projection_and_rope_arithmetic=True,
                pointer_table_and_duplicate_patch_bytes_charged=True,
                reference_constructor="direct_consolidated",
                candidate_constructor="direct_batched_checks",
                batched_check_ablation=True,
                original_bit_check_comparisons_per_layer=6,
                original_bit_verification_syncs_per_seal=1,
                maximum_segmented_final_ratio=1.01,
                maximum_segmented_peak_ratio=1.01,
                no_quantization=True,
                no_new_selection_or_benchmark_labels=True,
            )
        original(path, value)

    runtime.encoded.history.original_writer = register
    try:
        return previous.prepare(root, phase)
    finally:
        runtime.encoded.history.original_writer = original


def gpu_controls():
    from types import SimpleNamespace

    import torch

    from mosaickv.evaluation.encoded_projection import EncodedFormat
    from mosaickv.evaluation.encoded_projection_kernel import checkpoints
    from mosaickv.evaluation.exact_storage import SparseBitPatch, encode
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.projection_archive import ProjectionArchive
    from mosaickv.evaluation.segmented_history import make_tables
    from mosaickv.evaluation.segmented_history_kernel import fill

    torch.manual_seed(20261030)
    attention = SimpleNamespace(
        head_dim=128,
        k_proj=torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    records = []
    for flags in ((False, False), (True, False), (True, True)):
        formats = []
        for raw in flags:
            source = torch.randn(1, 1, 16, 256, device="cuda", dtype=torch.bfloat16)
            body = (
                PackedBF16(tuple(source.shape), True, (source.clone(),)) if raw else encode(source)
            )
            union = torch.arange(16, device="cuda", dtype=torch.int32)
            archive = ProjectionArchive(
                body,
                union,
                union[None].expand(2, -1),
                16,
                torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16),
                torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16),
            )
            patch = SparseBitPatch(
                torch.tensor([0, 16 * 128 + 129], device="cuda", dtype=torch.int32),
                torch.tensor([1.25, -3.5], device="cuda", dtype=torch.bfloat16),
            )
            formats.append(
                EncodedFormat(
                    "encoded_projection",
                    (1, 2, 16, 128),
                    archive=archive,
                    output_slots=union[None].expand(2, -1),
                    patches=(patch, patch),
                    literal_checkpoints=checkpoints(body),
                )
            )
        reference = tuple(
            torch.full((1, 2, 49, 128), -7, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        )
        actual = tuple(t.clone() for t in reference)
        for i, f in enumerate(formats):
            f.fill(
                attention,
                tuple(t[..., 5 + i * 16 : 5 + (i + 1) * 16, :] for t in reference),
                direct=True,
            )
        table = make_tables(tuple(formats), heads=2)
        fill(table, attention, tuple(t[..., 5:37, :] for t in actual))
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(reference, actual, strict=True)
        ):
            raise RuntimeError(
                "segmented projection differs from original packets "
                "or overwrites surrounding history"
            )
        records.append(
            dict(raw_flags=flags, original_KV_bits=True, strided_lease_sentinels_unchanged=True)
        )
    return records


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import torch

    from mosaickv.evaluation.storage_runs import run

    write_once(out / "segmented_gpu_controls.json", gpu_controls())
    if plan["stage"] == "canary":
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        positions = tuple(
            tuple(tuple(range(k.shape[-2])) for _ in range(k.shape[1])) for k, _ in source
        )
        variants, _ = runtime.encoded.fused_choices(
            adapter, prefill, capture, positions, compact=True
        )
        formats = next(iter(variants.values()))
        outputs, records = {}, {}
        for name, kwargs in (
            ("consolidated", dict(consolidated_history=True, source_history_constructor="direct")),
            (
                "checks_only",
                dict(consolidated_history=True, source_history_constructor="direct_batched_checks"),
            ),
            (
                "segmented",
                dict(segmented_history=True, source_history_constructor="direct_batched_checks"),
            ),
        ):
            outputs[name], records[name] = run(
                adapter,
                prefill,
                formats,
                tokens=128,
                audit=True,
                source_history=True,
                **kwargs,
            )
        passed = all(
            torch.equal(outputs["consolidated"], x) for x in outputs.values()
        ) and torch.equal(outputs["segmented"][:32], full_reference)
        write_once(
            out / "full_retention_segmented.json", dict(bitwise_128_logits=passed, records=records)
        )
        torch.save(outputs, out / "full_retention_segmented_logits.pt")
        if not passed:
            raise RuntimeError("segmented full-retention original logits differ")
    return runtime.study(
        adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan
    )


def admit(root, expected):
    plan = base.common.inputs(root, expected)
    record = base.common.read(root / "cpu_record.json")
    xml = Path(record["cpu_xml"])
    if (
        record["returncode"] != 0
        or record["source_sha"] != plan["source_sha"]
        or record["cpu_xml_sha256"] != sha(xml)
    ):
        raise ValueError("complete immutable-source CPU proof required")
    doc = ET.parse(xml).getroot()
    suites = list(doc.iter("testsuite"))
    count = sum(int(s.get("tests", "0")) for s in suites)
    if (
        count < 2709
        or count != len(list(doc.iter("testcase")))
        or any(int(s.get(k, "0")) for s in suites for k in ("errors", "failures", "skipped"))
    ):
        raise ValueError("full CPU suite must pass before GPU admission")
    from control_a100_schedule import ACTIVE, gpu_count, read_queue

    if sum(gpu_count(r["gres"]) for r in read_queue() if r["state"] in ACTIVE) > 4:
        raise ValueError("two canaries would exceed six aggregate A100s")
    write_once(
        root / "cpu_admission.json",
        dict(source_sha=plan["source_sha"], tests=count, cpu_xml_sha256=sha(xml)),
    )
    base.launch.submit(
        root,
        expected,
        script_name="evicastkv_segmented_history.sbatch",
        worker_minutes=30,
        memory_gb=24,
    )


if __name__ == "__main__":
    runtime.encoded.history.paired.install()
    runtime.prepare = prepare
    base.BASE, base.SCRIPT_NAME = ROOT, "evicastkv_segmented_history.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, study, runtime.analyze
    if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "admit"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", choices=("cpu", "admit"))
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--plan-sha", required=True)
        args = parser.parse_args()
        if args.mode == "cpu":
            previous.cpu(args.root, args.plan_sha)
        else:
            admit(args.root, args.plan_sha)
    else:
        base.main()
