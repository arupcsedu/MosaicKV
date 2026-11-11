#!/usr/bin/env python3
"""Synthetic native-dispatch diagnosis after the negative V23 operator gate.

No checkpoint, source packet, benchmark identity, or precision flag is changed.
All alternatives and failed rows are retained; this does not authorize models.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
POLICY = dict(
    id="native_BF16_bounded_dispatch_synthetic_v231",
    parent_source="fd23d9542f36c036e1286d6072cb2e2edafdff2c",
    seeds=[20261111, 13],
    lengths=[
        98,
        375,
        600,
        601,
        604,
        605,
        615,
        616,
        623,
        624,
        631,
        632,
        663,
        664,
        671,
        672,
        695,
        696,
        703,
        704,
        711,
        712,
        719,
        720,
        727,
        728,
        731,
    ],
    modes=[
        "group2",
        "group4",
        "group8",
        "group16",
        "broadcast32",
        "broadcast32_query_probability_clone",
    ],
    reference="native_BF16_B32_full_token_bmm",
    precision_flags_changed=False,
    numerical_gate="all_stage_words_identical",
    model_inference_authorized=False,
    whole_request_certification=False,
)


def compare(left, right):
    return dict(
        equal=torch.equal(left.view(torch.int16), right.view(torch.int16)),
        unequal=int((left.view(torch.int16) != right.view(torch.int16)).sum()),
        maximum_absolute_difference=float((left.float() - right.float()).abs().max()),
    )


@torch.inference_mode()
def run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    revision = clean_source(SOURCE)
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("one deterministic A100, CUBLAS config before Python required")
    torch.use_deterministic_algorithms(True)
    write_once(
        root / "manifest.json",
        dict(
            policy=POLICY,
            source_sha=revision,
            torch_version=torch.__version__,
            CUDA=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        ),
    )
    rows = []
    start = time.monotonic()
    for seed in POLICY["seeds"]:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        for length in POLICY["lengths"]:
            q = torch.randn((32, 1, 128), device="cuda", generator=generator).bfloat16()
            k, v = (
                torch.randn((32, length, 128), device="cuda", generator=generator).bfloat16()
                for _ in range(2)
            )
            s = torch.bmm(q, k.transpose(1, 2)) * 128**-0.5
            p = torch.softmax(s, -1, dtype=torch.float32).bfloat16()
            y = torch.bmm(p, v)
            for mode in POLICY["modes"]:
                actual_s, actual_y = torch.empty_like(s), torch.empty_like(y)
                if mode.startswith("group"):
                    batch = int(mode[5:])
                    for first in range(0, 32, batch):
                        sl = slice(first, first + batch)
                        ks, vs = k[sl].clone(), v[sl].clone()
                        scores = torch.bmm(q[sl], ks.transpose(1, 2)) * 128**-0.5
                        probs = torch.softmax(scores, -1, dtype=torch.float32).bfloat16()
                        actual_s[sl].copy_(scores)
                        actual_y[sl].copy_(torch.bmm(probs, vs))
                    kv_workspace = batch * length * 128 * 4
                else:
                    for head in range(32):
                        # The exact same head is functionally read in every
                        # native batch lane. Zero strides do not allocate KV.
                        keys = k[head : head + 1].clone().expand(32, -1, -1)
                        values = v[head : head + 1].clone().expand(32, -1, -1)
                        queries = q[head : head + 1].expand(32, -1, -1)
                        if mode.endswith("clone"):
                            queries = queries.clone()
                        scores = torch.bmm(queries, keys.transpose(1, 2)) * 128**-0.5
                        probs = torch.softmax(scores, -1, dtype=torch.float32).bfloat16()
                        if not mode.endswith("clone"):
                            probs = probs[:1].expand(32, -1, -1)
                        actual_s[head].copy_(scores[0])
                        actual_y[head].copy_(torch.bmm(probs, values)[0])
                    kv_workspace = length * 128 * 4
                scores_record, output_record = compare(s, actual_s), compare(y, actual_y)
                rows.append(
                    dict(
                        seed=seed,
                        length=length,
                        mode=mode,
                        scores=scores_record,
                        output=output_record,
                        all_equal=scores_record["equal"] and output_record["equal"],
                        maximum_functional_KV_workspace_bytes=kv_workspace,
                    )
                )
    summary = {
        mode: dict(
            all_equal=all(r["all_equal"] for r in rows if r["mode"] == mode),
            failed_rows=sum(not r["all_equal"] for r in rows if r["mode"] == mode),
            score_words=sum(r["scores"]["unequal"] for r in rows if r["mode"] == mode),
            output_words=sum(r["output"]["unequal"] for r in rows if r["mode"] == mode),
        )
        for mode in POLICY["modes"]
    }
    write_once(
        root / "result.json",
        dict(
            source_sha=revision,
            manifest_sha256=sha(root / "manifest.json"),
            rows=rows,
            summary=summary,
            wall_seconds=time.monotonic() - start,
            model_inference_authorized=False,
            gate_unchanged=True,
        ),
    )
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
