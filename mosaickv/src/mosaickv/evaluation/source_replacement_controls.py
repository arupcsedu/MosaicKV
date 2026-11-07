"""Bounded actual-CUDA controls, run before loading any checkpoint."""

from dataclasses import replace
from types import SimpleNamespace

import torch


@torch.inference_mode()
def gpu_controls():
    from mosaickv.evaluation.conditional_precision import pack_unprotected
    from mosaickv.evaluation.joint_incoming_projection import _decode_joint
    from mosaickv.evaluation.lt_source_chunks import decode_slice
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction, project
    from mosaickv.evaluation.source_hotpath import use as hotpath
    from mosaickv.evaluation.source_replacement import arguments, compact

    results = []
    for n in (17, 129, 603):
        width, heads, dim = 512, 4, 128
        x = torch.randn(1, 1, n, width, device="cuda", dtype=torch.bfloat16)
        omitted = torch.arange(n) % 3 == 0
        body = pack_unprotected(x, omitted.to("cuda")[None], bits=3, axis=2)
        compacted, evidence = compact(body, omitted)
        dense = SharedSourceFormat(
            body,
            (1, heads, n, dim),
            torch.empty(0, dtype=torch.int32, device="cuda"),
            (),
            torch.ones(1, n, dim, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, n, dim, device="cuda", dtype=torch.bfloat16),
            torch.arange(n, device="cuda", dtype=torch.int32),
            torch.arange(n, device="cuda", dtype=torch.int16)[None].expand(heads, n),
        )
        packed = replace(dense, source=compacted)
        expected = decode_source_for_construction(dense)
        actual = decode_source_for_construction(packed)
        if not torch.equal(expected.view(torch.int16), actual.view(torch.int16)):
            raise RuntimeError("compacted source CUDA decoding changed a BF16 word")
        for first in range(0, n, 128):
            take = min(128, n - first)
            a = torch.full((128, width), -99, dtype=torch.bfloat16, device="cuda")
            b = a.clone()
            decode_slice(dense, a, first, take)
            decode_slice(packed, b, first, take)
            if not torch.equal(a.view(torch.int16), b.view(torch.int16)):
                raise RuntimeError("native source slice or sentinel changed")
        incoming = torch.randn(width, dtype=torch.bfloat16, device="cuda")
        outputs = []
        tail = n % 128
        for packet in (dense, packed):
            out = torch.full((128, width), -99, dtype=torch.bfloat16, device="cuda")
            row_map, rows, mapped = arguments(packet.source)
            _decode_joint[((tail + 1 + 15) // 16, width // 64)](
                packet.source.codes,
                packet.source.minimum,
                packet.source.scale,
                packet.source.codes,
                packet.source.codes,
                incoming,
                out,
                n,
                row_map,
                FIRST=n - tail,
                TAKE=tail,
                WIDTH=width,
                HAS_COLUMNS=False,
                SOURCE_ROWS=rows,
                HAS_ROW_MAP=mapped,
                enable_fp_fusion=False,
            )
            outputs.append(out)
        if not torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)):
            raise RuntimeError("joint incoming-source reconstruction changed bits")
        attention = SimpleNamespace(
            k_proj=torch.nn.Linear(width, width, bias=False, device="cuda", dtype=torch.bfloat16),
            v_proj=torch.nn.Linear(width, width, bias=False, device="cuda", dtype=torch.bfloat16),
        )
        for group in (1, 4):
            for component in (0, 1):
                outputs = []
                for packet in (dense, packed):
                    target = torch.full(
                        (1, heads, n + 3, dim), -99, device="cuda", dtype=torch.bfloat16
                    )
                    with hotpath("prepared_restore"):
                        for _ in range(2):
                            project(
                                replace(packet, source_head_group=group),
                                attention,
                                (target, target),
                                component=component,
                            )
                    outputs.append(target)
                if not torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)):
                    raise RuntimeError("fused projection/prepared replay changed output bits")
        results.append(dict(rows=n, bitwise=True, **evidence))
    torch.cuda.synchronize()
    return dict(
        passed=True,
        cases=results,
        synthetic_operator_controls=True,
        no_benchmark_quality_claim=True,
    )
