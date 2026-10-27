#!/usr/bin/env python3
"""CPU dictionary/mass feasibility on two consumed captures, NOT VLM quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from mosaickv.prototypes.attention_compaction import (
    AttentionCompactionPolicy,
    fit_batched_mass_bias,
    plan_compaction_capacity,
)

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "mosaickv/configs/experiments/attention_compaction_v1.yaml"
SCOPE = (
    "online_compactor_enabled",
    "gpu_execution_authorized",
    "fresh_cohort_authorized",
    "calibration_authorized",
    "paper_campaign_authorized",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(protocol):
    if any(protocol[k] is not False for k in SCOPE):
        raise ValueError("CPU compaction audit cannot enable online/GPU/experiment scope")
    if (
        protocol["effective_method"] != "attention_matching_protected_reimpl"
        or protocol["classification"] != "cpu_consumed_capture_feasibility_not_benchmark_evidence"
        or protocol["dictionary_audit"]
        != dict(
            reference_queries=32,
            protected_positions=["first", "last"],
            selection="stable_mean_draft_attention_topk",
            maximum_cpu_threads=4,
            may_read_original_values=False,
            may_read_benchmark_annotations=False,
        )
    ):
        raise ValueError("unregistered compaction-audit method or evidence scope")
    policy = AttentionCompactionPolicy(**protocol["policy"])
    if not policy.enabled:
        raise ValueError("algebraic solver must be explicitly enabled for this CPU audit")
    return policy, plan_compaction_capacity(**protocol["capacity_planning"])


def audit_layer(data, slots, policy):
    """No K/V, outputs, annotations, model loader, or candidate forwards used."""
    import torch

    p, h = data["prompt_probabilities"], data["generated_mass"]
    if (
        p.ndim != 3
        or p.shape[:2] != (32, 32)
        or h.shape != (32, 32)
        or p.shape[-1] < slots
        or slots < 3
        or p.device.type != "cpu"
        or h.device.type != "cpu"
        or not torch.isfinite(p).all()
        or not torch.isfinite(h).all()
        or (p < 0).any()
        or (h < 0).any()
        or (p.sum(-1) + h > 1 + 1e-5).any()
    ):
        raise ValueError("audit requires complete finite CPU LLaVA MHA draft mass evidence")
    scores = p.double().mean(1)
    scores[:, 0], scores[:, -1] = -float("inf"), -float("inf")
    optional = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, : slots - 2]
    edge = torch.tensor([0, p.shape[-1] - 1]).expand(32, 2)
    indices = torch.cat((optional, edge), dim=-1).sort(-1).values
    attention = p.gather(2, indices[:, None].expand(-1, 32, -1)).double()
    protected = (indices == 0) | (indices == p.shape[-1] - 1)
    bias = fit_batched_mass_bias(attention, h, protected, policy)
    weights = bias.double().exp()
    before = (attention.sum(-1) + h - 1).square().mean(-1).sqrt()
    after = ((attention * weights[:, None]).sum(-1) + h - 1).square().mean(-1).sqrt()
    mutable = attention[:, :, 1:-1]
    singular = torch.linalg.svdvals(mutable)
    ranks = (singular > singular[:, :1] * policy.svd_relative_tolerance).sum(-1)
    ratios = singular[:, -1] / singular[:, 0].clamp_min(1e-30)
    return [
        dict(
            kv_head=head,
            prompt_source_positions=p.shape[-1],
            selected_positions=indices[head].tolist(),
            protected_stand_ins_only=True,
            baseline_mass_rmse=float(before[head]),
            fitted_mass_rmse=float(after[head]),
            numerical_rank=int(ranks[head]),
            reference_queries=32,
            smallest_largest_singular_ratio=float(ratios[head]),
            maximum_multiplier=float(weights[head].max()),
            protected_bias_zero=bool((bias[head][protected[head]] == 0).all()),
        )
        for head in range(32)
    ]


def run(output, protocol_path=PROTOCOL):
    import torch

    scratch = Path("/scratch/djy8hg").resolve()
    if output.exists() or scratch not in output.resolve().parents:
        raise ValueError("CPU audit requires a NEW immutable directory under /scratch/djy8hg")
    # Unlike exploratory tests, the saved-evidence audit requires a clean SHA.
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True):
        raise ValueError("commit the scoped compaction implementation before canonical CPU audit")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    protocol = yaml.safe_load(protocol_path.read_text())
    policy, capacity = validate_protocol(protocol)
    parent = Path(protocol["consumed_capture_root"])
    if scratch not in parent.resolve().parents:
        raise ValueError("consumed captures must remain under scratch")
    bound = {str(protocol_path): digest(protocol_path)}
    for capture in protocol["captures"]:
        folder = parent / "workers" / str(capture["worker"])
        for name, expected in (
            ("draft-evidence.pt", capture["evidence_sha256"]),
            ("manifest.json", capture["manifest_sha256"]),
        ):
            path = folder / name
            if digest(path) != expected:
                raise ValueError(f"consumed input hash mismatch: {path}")
            bound[str(path)] = expected
    torch.set_num_threads(protocol["dictionary_audit"]["maximum_cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started, rows, producers = time.perf_counter(), [], []
    for capture in protocol["captures"]:
        folder = parent / "workers" / str(capture["worker"])
        producer = json.loads((folder / "manifest.json").read_text())
        producers.append(
            dict(worker=capture["worker"], model=producer["model"], dataset=producer["dataset"])
        )
        layers = torch.load(folder / "draft-evidence.pt", map_location="cpu", weights_only=True)
        if not isinstance(layers, tuple) or len(layers) != 32:
            raise ValueError("complete 32-layer saved evidence required")
        for layer, data in enumerate(layers):
            rows.extend(
                dict(worker=capture["worker"], layer=layer, **row)
                for row in audit_layer(data, capacity.prompt_positions, policy)
            )
    for path, expected in bound.items():
        if digest(path) != expected:
            raise ValueError("input changed during CPU analysis")
    summary = dict(
        measurement_type="cpu_saved_capture_algebraic_feasibility",
        source_sha=sha,
        config_sha=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        source_worktree_dirty=False,
        bound_files=bound,
        input_producers=producers,
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda="not_used",
            driver="not_used",
            gpu_type="not_used",
            gpu_count=0,
            transformers="not_used",
            vllm="not_used",
            sglang="not_used",
            backend="cpu_algebra",
            attention_implementation="not_used",
            seed=0,
        ),
        capacity_planning_not_measured=asdict(capacity),
        routes=len(rows),
        mean_mass_rmse_before=sum(r["baseline_mass_rmse"] for r in rows) / len(rows),
        mean_mass_rmse_after=sum(r["fitted_mass_rmse"] for r in rows) / len(rows),
        full_row_rank_routes=sum(r["numerical_rank"] == 32 for r in rows),
        elapsed_seconds=time.perf_counter() - started,
        original_values_available=False,
        value_norm_safety_measured=False,
        candidate_model_output_measured=False,
        benchmark_quality_measured=False,
        new_identities=0,
        limitations=[
            "Synthetic first/last protection is not the real mandatory-token map.",
            "FullKV reference queries are not compressed-model queries.",
            "Rank and mass fit do not establish value-fit safety or answer quality.",
            "Capacity arithmetic excludes transient construction/model workspaces.",
        ],
        **dict.fromkeys(SCOPE, False),
    )
    output.mkdir(parents=True)
    with (output / "routes.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    summary["routes_sha256"] = digest(output / "routes.jsonl")
    with (output / "report.json").open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "routes",
                    "mean_mass_rmse_before",
                    "mean_mass_rmse_after",
                    "full_row_rank_routes",
                    "elapsed_seconds",
                    "benchmark_quality_measured",
                )
            },
            sort_keys=True,
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
