#!/usr/bin/env python3
"""Common private-prefill score reuse: FullKV identity then consumed cache parity."""

from pathlib import Path
from unittest.mock import patch

import early_scale_header_study as early
import exact_scale_replay_study as base
import selective_history_study as precision

from mosaickv.evaluation.fox_campaign import write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/mixed-verification-barrier-20261108-v1")
ARMS = ["private_score_reference", "private_score_reuse"]
base.ARMS = ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/private_prefill_scores.sbatch"
base.EXTRA_TESTS = (
    *base.EXTRA_TESTS,
    "test_fused_history_unpack.py",
    "test_mixed_history_projection.py",
    "test_paired_history_source.py",
    "test_mixed_verification_barrier.py",
    "test_private_prefill_scores.py",
    "test_tiled_softmax.py",
    "test_private_prefill_studies.py",
)
original_analyze = base.analyze


def prepare(root):
    if not base.common.read(PARENT / "analysis/verification-decision.json")[
        "bitwise_and_persistent_storage_gate"
    ]:
        raise ValueError("complete original epoch-verification parity required")
    early.PARENT = PARENT
    early.prepare(
        root,
        dependencies=(PARENT / "analysis/verification-decision.json",),
        overrides=dict(
            study="common_private_prefill_score_reuse_v1",
            measurement_type="consumed_common_private_prefill_score_reuse",
            arms=ARMS,
            mixed_history_projection=(
                "unchanged complete reader; share private eager score/probability buffer"
            ),
        ),
    )


def operator(root, p, folder, i):
    from types import SimpleNamespace

    import torch
    from compact_hybrid_study import manifest
    from packed_history_study import trials
    from transformers.models.llama import modeling_llama as llama

    from mosaickv.evaluation.tiled_softmax import use

    manifest(p, base.sha(root / "plan.json"), folder, i, gpu=True)
    module = SimpleNamespace(training=False, num_key_value_groups=1)
    records = []
    with torch.inference_mode():
        for length, keys in ((17, 63), (603, 603), (1, 731)):
            q, k, v = (
                torch.randn(1, 32, n, 128, device="cuda").bfloat16() for n in (length, keys, keys)
            )
            mask = torch.zeros(1, 1, length, keys, device="cuda", dtype=torch.bfloat16)
            mask[:, :, :, -1] = float("-inf")
            original = tuple(x.clone() for x in (q, k, v, mask))
            with use(rows=p["tiled_softmax_rows"]):
                expected = llama.eager_attention_forward(module, q, k, v, mask, 128**-0.5)
            for flag in (True, False, True):
                with use(rows=p["tiled_softmax_rows"], inplace=flag):
                    actual = llama.eager_attention_forward(module, q, k, v, mask, 128**-0.5)
                if not all(
                    torch.equal(a.view(torch.int16), b.view(torch.int16))
                    for a, b in zip(actual, expected, strict=True)
                ):
                    raise RuntimeError(
                        "original complete attention output/probabilities changed a BF16 word"
                    )
                if not all(a.equal(b) for a, b in zip(original, (q, k, v, mask), strict=True)):
                    raise RuntimeError("caller-owned attention input was modified")
            timings = {}
            for flag in (False, True):
                with use(rows=p["tiled_softmax_rows"], inplace=flag):
                    timings[str(flag)] = trials(
                        lambda q=q, k=k, v=v, mask=mask: llama.eager_attention_forward(
                            module, q, k, v, mask, 128**-0.5
                        ),
                        p,
                    )
            records.append(
                dict(
                    query_length=length,
                    key_length=keys,
                    complete_output_and_probabilities_bitwise=True,
                    caller_input_words_unchanged=True,
                    operator_trials_NOT_full_request_latency=timings,
                )
            )
    write_once(
        folder / "result.json",
        dict(status="completed", passed=True, controls=records, pretrained_weights_loaded=False),
    )


def model(root, p, folder, i):
    import evicastkv_paper_precision as campaign
    import torch

    from mosaickv.adapters import huggingface as hf
    from mosaickv.evaluation.early_scale_headers import use as early_headers
    from mosaickv.evaluation.exact_scale_replay import use as late_headers
    from mosaickv.evaluation.fused_history_unpack import use as fused
    from mosaickv.evaluation.history_source_precision import use as history
    from mosaickv.evaluation.mixed_history_projection import use as mixed
    from mosaickv.evaluation.mixed_verification_barrier import use as barrier
    from mosaickv.evaluation.paired_history_source import use as pairs
    from mosaickv.evaluation.tiled_softmax import use as tile

    reference = []

    def context(stack, arm):
        return dict(
            early=stack.enter_context(early_headers()),
            late=stack.enter_context(late_headers()),
            selective_history=stack.enter_context(
                history(feature_group=32, exact_recent_tokens=64)
            ),
            mixed_projection=stack.enter_context(mixed()),
            fused_unpack=stack.enter_context(fused()),
            paired_source=stack.enter_context(pairs()),
            verification_barrier=stack.enter_context(barrier()),
            private_scores=stack.enter_context(
                tile(rows=p["tiled_softmax_rows"], inplace=arm == ARMS[1])
            ),
        )

    def compare(actual, parent, plan, arm):
        proof = precision.fidelity(actual, parent, plan, precision.ARMS[1])
        if arm == ARMS[0]:
            reference.append(actual.clone())
        if not reference:
            raise ValueError("original same-precision prefill reference required")
        proof["all_128_logits_bitwise_equal_original"] = bool(torch.equal(actual, reference[0]))
        proof["passed"] &= proof["all_128_logits_bitwise_equal_original"]
        return proof

    # The common worker writes its manifest BEFORE loading the adapter. Inject
    # raw FullKV identity immediately after that load, reusing this same model.
    original_load = hf.load_hf_adapter

    def load(*args, **kwargs):
        adapter = original_load(*args, **kwargs)
        sample = base.common.read(p["controls"]["inputs"])[i]
        ordinary = []
        for flag in (False, True):
            with tile(rows=p["tiled_softmax_rows"], inplace=flag):
                logits, _ = campaign.execute(
                    adapter, sample, p["controls"]["parent"], "full_kv", audit=False
                )
            ordinary.append(logits.cpu())
            torch.save(ordinary[-1], folder / f"fullkv-private-{flag}.logits.pt")
        passed = bool(torch.equal(*ordinary))
        write_once(
            folder / "fullkv-identity.json",
            dict(
                passed=passed,
                all_128_logits_bitwise=passed,
                reference_logits_sha=base.sha(folder / "fullkv-private-False.logits.pt"),
                candidate_logits_sha=base.sha(folder / "fullkv-private-True.logits.pt"),
                retention_ratio=1.0,
                method="full_kv",
                no_pruning_merging_quantization_repair=True,
                manifest_path=str(folder / "manifest.json"),
                maximum_logit_difference=float((ordinary[0] - ordinary[1]).abs().max()),
                token_agreement=float(
                    (ordinary[0].argmax(-1) == ordinary[1].argmax(-1)).float().mean()
                ),
                latency_NOT_a_compressed_method_comparison=True,
            ),
        )
        if not passed:
            raise RuntimeError("private prefill score reuse failed FullKV identity")
        return adapter

    base.common.extra_model_context, base.common.assess_fidelity = context, compare
    with patch.object(hf, "load_hf_adapter", load):
        base.common.model(root, p, folder, i)


def analyze(root, p):
    original_analyze(root, p)
    proof = base.common.read(root / "analysis/decision.json")
    eligible = proof["complete"]
    pairs = []
    for i in (0, 1):
        full = (
            base.common.read(root / f"model/{i}/fullkv-identity.json")
            if (root / f"model/{i}/fullkv-identity.json").exists()
            else {}
        )
        rows = [r for r in proof["rows"] if r["input"] == i]
        if len(rows) != 2 or any(r["status"] != "completed" for r in rows):
            eligible = False
            continue
        a, b = rows
        info = b.get("common_header", {}).get("private_scores") or {}
        good = bool(
            full.get("passed")
            and b["fidelity"].get("all_128_logits_bitwise_equal_original")
            and a["initial_GPU_KV_bytes"] == b["initial_GPU_KV_bytes"]
            and a["final_GPU_KV_bytes"] == b["final_GPU_KV_bytes"]
            and info.get("private_score_buffer_reused")
            and info.get("calls", 0) >= 32
            and isinstance(a["whole_request_KV_plus_workspace_peak_bytes"], int)
            and isinstance(b["whole_request_KV_plus_workspace_peak_bytes"], int)
            and b["whole_request_KV_plus_workspace_peak_bytes"]
            < a["whole_request_KV_plus_workspace_peak_bytes"]
        )
        eligible &= good
        pairs.append(dict(input=i, gate=good, FullKV_identity=full, private_scores=info))
    write_once(
        root / "analysis/private-score-decision.json",
        dict(
            complete=proof["complete"],
            bitwise_and_peak_gate=bool(eligible),
            pairs=pairs,
            fresh_cohort_authorized=False,
            independent_quality_established=False,
            complete_superiority_established=False,
        ),
    )
    # Compatible baselines must receive the common optimization too. Their
    # numerical controls are a separate finite successor, never new labels.
    if eligible:
        from private_prefill_baselines_study import prepare as prepare_baselines

        prepare_baselines(root / "compatible-baselines")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
