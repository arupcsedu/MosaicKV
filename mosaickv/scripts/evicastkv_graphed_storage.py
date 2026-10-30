#!/usr/bin/env python3
"""Finite bit-exact shared-workspace/capture canary and six-path comparison."""

from pathlib import Path

import evicastkv_source_history_runtime as runtime

from mosaickv.evaluation.fox_campaign import sha

base = runtime.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/graphed-storage-consumed-20261030-v3")
VALIDATED = Path(
    "/scratch/djy8hg/runs/evicastkv/source-history-direct-consumed-20261030-v1/comparison"
)
original_prepare = runtime.prepare


def prepare(root, phase):
    gate = base.common.read(VALIDATED / "analysis/source_history_decision.json")
    if not gate["runtime_canary_passed"]:
        raise ValueError("final direct source-history controls required")
    original = runtime.encoded.history.original_writer

    def register(path, value):
        if path.name == "plan.json":
            for name in [
                "plan.json",
                "analysis/GATE.json",
                "analysis/manifest.json",
                "analysis/source_history_decision.json",
            ]:
                bound = VALIDATED / name
                value["bound_files"][str(bound)] = sha(bound)
            value.update(
                protocol="shared_reconstruction_workspace_graphs_original_HF_attention_v1",
                source_history_comparison_kind="workspace",
                candidate_constructor="direct",
                reference_constructor="direct",
                maximum_workspace_peak_ratio=1.05,
                graphs_capture_immutable_reconstruction_only=True,
                source_projection_weights_and_attention_unchanged=True,
                strict_bitwise_original_logits_required=True,
                full_retention_controls_128_tokens=root.name == "canary",
                new_attention_kernel_rejected_not_used=True,
                whole_request_peak_certified=False,
            )
        original(path, value)

    runtime.encoded.history.original_writer = register
    try:
        return original_prepare(root, phase)
    finally:
        runtime.encoded.history.original_writer = original


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import torch

    from mosaickv.evaluation.graphed_storage import gpu_controls
    from mosaickv.evaluation.storage_runs import run

    runtime.write_once(out / "shared_workspace_gpu_controls.json", gpu_controls())

    if plan["stage"] == "canary":
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        full = tuple(tuple(tuple(range(k.shape[-2])) for _ in range(k.shape[1])) for k, _ in source)
        variants, _ = runtime.encoded.fused_choices(adapter, prefill, capture, full, compact=True)
        formats = next(iter(variants.values()))
        outputs, records = {}, {}
        for name, policy in [
            ("legacy", {}),
            ("workspace_eager", dict(shared_workspace=True)),
            ("workspace_graph", dict(shared_workspace=True, workspace_graph=True)),
        ]:
            outputs[name], records[name] = run(
                adapter,
                prefill,
                formats,
                tokens=128,
                source_history=True,
                source_history_constructor="direct",
                audit=True,
                **policy,
            )
        passed = all(torch.equal(outputs["legacy"], tensor) for tensor in outputs.values())
        passed = passed and torch.equal(outputs["legacy"][:32], full_reference)
        write_once = runtime.write_once
        write_once(
            out / "full_retention_workspace_controls.json",
            dict(
                original_128_logits_bitwise=passed,
                records=records,
                maximum_logit_difference=max(
                    float((tensor - outputs["legacy"]).abs().max()) for tensor in outputs.values()
                ),
            ),
        )
        torch.save(outputs, out / "full_retention_workspace_logits.pt")
        if not passed:
            raise RuntimeError("shared workspace/graph changed original retention-one logits")
    return runtime.study(
        adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan
    )


if __name__ == "__main__":
    runtime.encoded.history.paired.install()
    runtime.prepare = prepare
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_graphed_storage.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, study, runtime.analyze
    base.main()
