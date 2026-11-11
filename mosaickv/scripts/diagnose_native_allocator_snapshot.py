"""No-weight, consumed-input-free capture of allocator schema violations."""

import argparse
import os
from collections import Counter
from pathlib import Path

import torch

from mosaickv.evaluation.bounded_component_attention import selftest
from mosaickv.evaluation.fox_campaign import clean_source, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    revision = clean_source(Path(__file__).resolve().parents[2])
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise ValueError("one allocated A100 required")
    torch.use_deterministic_algorithms(True)
    arithmetic = selftest(device="cuda")
    small = [torch.empty(2 * (1 + i % 63), dtype=torch.uint8, device="cuda") for i in range(1000)]
    sources = [torch.empty((n, 4096), dtype=torch.bfloat16, device="cuda") for n in (600, 604)]
    metadata = torch.arange(1000, device="cuda")
    noise = [torch.empty(1, dtype=torch.uint8, device="cuda") for _ in range(7000)]
    owners = [
        (x.untyped_storage().data_ptr(), x.untyped_storage().nbytes())
        for x in (*small, *sources, metadata)
    ]
    snapshot = torch.cuda.memory_snapshot()
    violations, states = [], Counter()
    for segment in snapshot:
        cursor = segment["address"]
        for block in segment["blocks"]:
            address = block.get("address", cursor)
            cursor = address + block["size"]
            states[block["state"]] += 1
            if block.get("requested_size", 0) > block["size"]:
                violations.append(
                    dict(
                        address=address,
                        size=block["size"],
                        requested_size=block["requested_size"],
                        state=block["state"],
                        intersects_required_owner=any(
                            p < cursor and p + n > address for p, n in owners
                        ),
                        is_expandable=segment.get("is_expandable"),
                    )
                )
    args.root.mkdir(parents=True, exist_ok=False)
    write_once(args.root / "snapshot.json", dict(segments=snapshot, required_owners=owners))
    write_once(
        args.root / "diagnosis.json",
        dict(
            source_sha=revision,
            SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
            torch_version=torch.__version__,
            backend=torch.cuda.memory.get_allocator_backend(),
            allocator_env=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            arithmetic_pass=arithmetic["all_bitwise_passed"],
            violations=violations,
            state_counts=dict(states),
            nonKV_noise_allocations=len(noise),
            no_model_weights=True,
            no_numerical_guard_changed=True,
        ),
    )


if __name__ == "__main__":
    main()
