"""Native-style fused tile plus fixed exact repairs, paired graph operator only."""

import random
import time
from types import SimpleNamespace

import torch

from mosaickv.evaluation.row_major_source_operator import summarize

PATCH_CAP = 32768


def validate_patches(patches):
    if type(patches) is not tuple or len(patches) != 2:
        raise ValueError("both frozen component repairs required")
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    size = backing_bytes(tuple(t for p in patches for t in p.tensors))
    if size > PATCH_CAP:
        raise RuntimeError(f"native tile exceeded {PATCH_CAP} repair bytes: {size}")
    return size


@torch.inference_mode()
def run(index):
    from mosaickv.evaluation.exact_storage import SparseBitPatch
    from mosaickv.evaluation.full_prompt_source import decode_into
    from mosaickv.evaluation.lossless_affine_minimum import encode_minimum
    from mosaickv.evaluation.precision_tier import pack
    from mosaickv.evaluation.prompt_graph_reader import allocator_blocks, generator_metadata
    from mosaickv.evaluation.repaired_source_tile import decode_control, project_raw
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat, apply_transport_patch
    from mosaickv.evaluation.shared_source_kernel import project
    from mosaickv.evaluation.source_channel_exceptions import ChannelExceptions

    if type(index) is not int or index not in (0, 1):
        raise ValueError("two frozen consumed geometries only")
    n, h, d = (600, 604)[index], 32, 128
    generator = torch.Generator(device="cuda").manual_seed(20261104)
    x = torch.randn((1, 1, n, h * d), generator=generator, device="cuda").bfloat16()
    phase = torch.linspace(-3, 3, n * 64, device="cuda").reshape(1, n, 64)
    lookup = torch.full((h * d,), 255, device="cuda", dtype=torch.uint8)
    chosen = torch.tensor([0, 31, h * d - 1], device="cuda")
    lookup[chosen] = torch.arange(3, device="cuda", dtype=torch.uint8)
    columns = ChannelExceptions(lookup, x[0, 0].index_select(1, chosen).T.contiguous().clone())
    packet = SharedSourceFormat(
        encode_minimum(pack(x, bits=3, axis=2)),
        (1, h, n, d),
        torch.empty(0, device="cuda", dtype=torch.int32),
        (),
        torch.cat((phase.cos(), phase.cos()), -1).bfloat16(),
        torch.cat((phase.sin(), phase.sin()), -1).bfloat16(),
        None,
        None,
        source_head_group=4,
        channel_exceptions=columns if index else None,
    )
    a, b = tuple(torch.empty((1, n, h * d), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    decode_into(packet, a)
    decode_control(packet, b)
    source_equal = torch.equal(a.view(torch.int16), b.view(torch.int16))
    if not source_equal:
        raise RuntimeError("native tile changed an original dequantized input word")
    del a, b
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    reference = tuple(
        torch.empty(packet.shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    raw = tuple(torch.empty_like(t) for t in reference)
    torch.cuda.synchronize()
    fit_start = time.perf_counter()
    resources = []
    for c in (0, 1):
        project(packet, attention, (reference[c], reference[c]), component=c)
        resources.append(project_raw(packet, attention, c, raw[c], inspect_kernel=True))
    patches = tuple(
        SparseBitPatch.build(ref, actual) for ref, actual in zip(reference, raw, strict=True)
    )
    patch_bytes = validate_patches(patches)
    torch.cuda.synchronize()
    fitting_wall_seconds = time.perf_counter() - fit_start
    counts = [p.indices.numel() for p in patches]
    del raw
    # Never refit for a changed destination stride or decode prefix.
    stride_controls = []
    for tail in (0, 1, 64, 127):
        targets = tuple(
            torch.full((1, h, n + tail, d), -71.0, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )
        for c in (0, 1):
            project_raw(packet, attention, c, targets[c])
            apply_transport_patch(patches[c], targets[c], prompt_length=n)
            if not torch.equal(
                targets[c][..., :n, :].view(torch.int16), reference[c].view(torch.int16)
            ) or not bool((targets[c][..., n:, :] == -71.0).all()):
                raise RuntimeError(
                    "fixed native tile repair changed a parent word or history sentinel"
                )
        stride_controls.append(dict(tail=tail, KV_bitwise=True, history_sentinels_unchanged=True))
        del targets
    actual = tuple(
        torch.full((1, h, n + 127, d), -71.0, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )

    def execute(candidate):
        for c in (0, 1):
            if candidate:
                project_raw(packet, attention, c, actual[c])
                apply_transport_patch(patches[c], actual[c], prompt_length=n)
            else:
                project(packet, attention, (actual[c], actual[c]), component=c)

    execute(False)
    execute(True)
    graphs, graph = {}, None
    torch.cuda.synchronize()
    before, blocks, stream = torch.cuda.memory_allocated(), allocator_blocks(), torch.cuda.Stream()
    records = []
    try:
        for mode in (False, True):
            graph = torch.cuda.CUDAGraph()
            current = torch.cuda.current_stream()
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    execute(mode)
                finally:
                    graph.capture_end()
            current.wait_stream(stream)
            graphs[mode] = graph
        torch.cuda.synchronize()
        addresses = generator_metadata(
            blocks, allocator_blocks(), torch.cuda.memory_allocated() - before
        )
        rng = random.Random(20261104)
        for trial in range(-5, 20):
            placement = [False, True]
            rng.shuffle(placement)
            for candidate in placement:
                for target in actual:
                    target.fill_(-71.0)
                torch.cuda.synchronize()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start = time.perf_counter()
                begin.record()
                graphs[candidate].replay()
                end.record()
                torch.cuda.synchronize()
                wall = time.perf_counter() - start
                equal = all(
                    torch.equal(t[..., :n, :].view(torch.int16), ref.view(torch.int16))
                    and bool((t[..., n:, :] == -71.0).all())
                    for t, ref in zip(actual, reference, strict=True)
                )
                if not equal:
                    raise RuntimeError("native tiled graph changed frozen parent bits")
                records.append(
                    dict(
                        trial=trial,
                        row4=candidate,
                        repaired_tile=candidate,
                        wall_seconds=wall,
                        cuda_interval_ms=begin.elapsed_time(end),
                        bitwise=True,
                        added_persistent_patch_bytes=patch_bytes if candidate else 0,
                        extra_workspace_backing_bytes=0,
                        registered_graph_metadata_bytes=1024,
                    )
                )
    finally:
        torch.cuda.synchronize()
        live = torch.cuda.memory_allocated()
        for graph in graphs.values():
            graph.reset()
        graphs.clear()
        del graph
        torch.cuda.synchronize()
        if live - torch.cuda.memory_allocated() != 1024:
            raise RuntimeError("native tile did not retire exact graph ownership")
    return dict(
        rows=records,
        summary=summarize(records),
        resources=resources,
        original_source_words_bitwise=source_equal,
        stride_controls=stride_controls,
        numerical_patch_bytes=patch_bytes,
        numerical_patch_words=counts,
        fitting_wall_seconds=fitting_wall_seconds,
        fitting_excluded_from_operator_replay_not_full_request_evidence=True,
        original_and_candidate_both_private_graph_replay=True,
        no_full_source_or_output_workspace=True,
        generator_addresses_count=len(addresses),
        generator_metadata_retired_bytes=1024,
        shape=[1, h, n, d],
        independent_quality_N=0,
        no_checkpoint_loaded=True,
        numerical_tolerance_not_relaxed=True,
        patch_fitted_to_original_decoder_not_labels=True,
    )
