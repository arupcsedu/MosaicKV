"""Small exact reader controls, no model weights or benchmark labels."""

from types import SimpleNamespace


def gpu_controls(policy):
    import torch
    import triton

    from mosaickv.evaluation import encoded_projection_kernel as encoded
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_bf16_kernel import encode
    from mosaickv.evaluation.precision_tier import pack
    from mosaickv.evaluation.reconstruction_policy import use, validate
    from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction

    validate(policy)
    records = []
    if policy in ("reuse_headers", "reuse_both"):
        generator = torch.Generator(device="cuda").manual_seed(20261102)
        for n in (1, 15, 16, 17, 127, 128, 129, 599, 600, 604):
            x = torch.randn((1, 1, n, 512), device="cuda", generator=generator).bfloat16()
            body = pack(x, bits=3, axis=2)
            p = SimpleNamespace(source=body, channel_exceptions=None)
            with use("original"):
                left = decode_source_for_construction(p)
            with use(policy):
                right = decode_source_for_construction(p)
            if not torch.equal(left.view(torch.int16), right.view(torch.int16)):
                raise RuntimeError("shared-header reuse changed a decoded source bit")
            records.append(dict(reader="headers", rows=n, bitwise=True))
    if policy in ("carry_literals", "reuse_both"):
        words = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        tensor = words.view(torch.bfloat16).reshape(1, 1, 256, 256)
        ordinary = encode(tensor, raw_fallback=False)
        bits = words.to(torch.int32) & 65535
        count = words.numel()
        literal = PackedBF16(
            tuple(tensor.shape),
            False,
            (
                ((bits & 127) | ((bits >> 8) & 128)).to(torch.uint8),
                torch.full((count // 8 * 3,), 255, dtype=torch.uint8, device="cuda"),
                torch.zeros(count // 1024, dtype=torch.uint8, device="cuda"),
                torch.arange(0, count + 1, 1024, dtype=torch.int32, device="cuda"),
                ((bits >> 7) & 255).to(torch.uint8),
            ),
        )
        for name, body in (("all_65536_patterns", ordinary), ("all_literal_overflow", literal)):
            check = encoded.checkpoints(body)
            output = torch.empty_like(tensor)
            encoded._materialize_carried[(triton.cdiv(256, 16),)](
                *encoded.pointers(body), check, output.view(torch.uint16), 256, WIDTH=256
            )
            if not torch.equal(output.view(torch.int16), tensor.view(torch.int16)):
                raise RuntimeError("carried literal counts changed a BF16 bit pattern")
            records.append(dict(reader="literals", case=name, bitwise=True))
    torch.cuda.synchronize()
    # Compile the actual projection loops too, not just the standalone readers.
    # These are synthetic numerical controls, never benchmark-quality rows.
    from mosaickv.evaluation import shared_source_kernel as shared

    n, width, heads = 17, 512, 4
    x = torch.randn((1, 1, n, width), device="cuda").bfloat16()
    attention = SimpleNamespace(
        head_dim=128,
        k_proj=torch.nn.Linear(width, width, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(width, width, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    cos = torch.ones((1, n, 128), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    source = SimpleNamespace(
        shape=(1, heads, n, 128), source=pack(x, bits=3, axis=2), cos=cos, sin=sin
    )
    body = encode(x, raw_fallback=False)
    check = encoded.checkpoints(body)
    union = torch.arange(n, device="cuda", dtype=torch.int32)
    slots = union.to(torch.int16)[None].expand(heads, -1)
    archive = SimpleNamespace(
        selected=SimpleNamespace(shape=(heads, n)), union=union, cos=cos, sin=sin
    )
    for kind in ("source", "encoded"):
        for component in (-1, 0, 1):
            outputs = []
            for label in ("original", policy):
                target = tuple(
                    torch.empty((1, heads, n, 128), device="cuda", dtype=torch.bfloat16)
                    for _ in range(2)
                )
                with use(label):
                    if kind == "source":
                        shared.project(source, attention, target, component=component)
                    else:
                        encoded.project(
                            body, check, attention, archive, slots, target, component=component
                        )
                outputs.append(target)
            for c in range(2) if component == -1 else (component,):
                if not torch.equal(
                    outputs[0][c].view(torch.int16), outputs[1][c].view(torch.int16)
                ):
                    raise RuntimeError("reader reuse changed projection output bits")
            records.append(dict(reader=kind, component=component, projection_bitwise=True))
    torch.cuda.synchronize()
    return records
