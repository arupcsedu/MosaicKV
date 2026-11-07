"""Finite CUDA execution-only screen; synthetic fixtures are not task results."""

import random
from time import perf_counter

import torch

from mosaickv.evaluation.guard_tail_graph import GuardTail, native_tail


def lifetime_control(norm, head, pair, capture_mode):
    """The guard must retire before the enclosing graph's RNG-owner audit."""
    from mosaickv.evaluation.post_attention_segment import capture_stream
    from mosaickv.evaluation.prompt_graph_reader import allocator_blocks, generator_metadata

    source = torch.ones(1, device="cuda")
    destination = torch.empty_like(source)
    current, stream = torch.cuda.current_stream(), capture_stream(source.device)
    torch.cuda.synchronize()
    before = allocator_blocks()
    allocated = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        graph.capture_begin()
        try:
            destination.copy_(source)
        finally:
            graph.capture_end()
    current.wait_stream(stream)
    torch.cuda.synchronize()
    addresses = generator_metadata(
        before, allocator_blocks(), torch.cuda.memory_allocated() - allocated
    )
    tail = GuardTail(norm, head, capture_mode=capture_mode)
    output, logits, record = tail.run(*pair)
    tail.close()
    del output, logits, record, tail
    torch.cuda.synchronize()
    if not all(address in allocator_blocks() for address in addresses):
        raise RuntimeError("guard prematurely retired the enclosing graph's RNG owners")
    allocated = torch.cuda.memory_allocated()
    graph.reset()
    del graph
    torch.cuda.synchronize()
    released = allocated - torch.cuda.memory_allocated()
    if released != 1024 or any(address in allocator_blocks() for address in addresses):
        raise RuntimeError("guard lifecycle violates the unchanged two-scalar retirement check")
    return dict(registered_scalar_bytes=1024, retired_scalar_bytes=released, passed=True)


@torch.inference_mode()
def run(*, capture_mode="context_manager"):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    norm = LlamaRMSNorm(4096).to(device="cuda", dtype=torch.bfloat16)
    head = torch.nn.Linear(4096, 32000, bias=False, device="cuda", dtype=torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(20261107)
    head.weight.copy_(torch.randn(head.weight.shape, generator=generator, device="cuda") * 0.01)
    pairs = []
    for delta in (0.0, 0.01, 0.2):
        parent = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16, generator=generator)
        pairs.append((parent, parent + delta))
    native = []
    for parent, active in pairs:
        native.append(tuple(x.clone() for x in native_tail(norm, head, parent, active)))
    g = GuardTail(norm, head, capture_mode=capture_mode)
    for index in (0, 1, 2, 1, 0):
        output, logits, record = g.run(*pairs[index])
        wanted = native[index]
        if not (
            torch.equal(output, wanted[0])
            and torch.equal(logits, wanted[1])
            and torch.equal(record, wanted[2][:4].cpu())
        ):
            raise RuntimeError("native guard-tail tensors/FP64 reductions differ bitwise")
    nan = pairs[0][0].clone().fill_(float("nan"))
    try:
        g.run(nan, nan)
    except ValueError as error:
        if "nonfinite" not in str(error):
            raise
    else:
        raise RuntimeError("nonfinite logits silently accepted")
    resources = g.report()
    g.close()
    del g, native, output, logits, record, nan
    lifetime = lifetime_control(norm, head, pairs[0], capture_mode)
    results = {"reference": [], "graph": []}
    rng = random.Random(20261107)
    # Each block is a new request, including capture, warmup and graph retirement.
    for trial in range(8):
        order = ["reference", "graph"]
        rng.shuffle(order)
        for arm in order:
            torch.cuda.synchronize()
            begin = perf_counter()
            graph = GuardTail(norm, head, capture_mode=capture_mode) if arm == "graph" else None
            for step in range(127):
                parent, active = pairs[step % len(pairs)]
                if graph is None:
                    out, logits, stats = native_tail(norm, head, parent, active)
                    record = stats.detach().cpu()
                    if not bool(record[4]):
                        raise RuntimeError("nonfinite synthetic reference")
                else:
                    out, logits, record = graph.run(parent, active)
                cpu_logits = logits.detach().cpu()
                del out, logits, record, cpu_logits
            if graph is not None:
                graph.close()
            torch.cuda.synchronize()
            elapsed = perf_counter() - begin
            if trial >= 2:
                results[arm].append(elapsed)
    ratio = sum(results["graph"]) / sum(results["reference"])
    return dict(
        classification="synthetic_native_operator_only_not_measured_model_quality",
        capture_mode=capture_mode,
        synthetic_weights=True,
        bitwise_native_controls=True,
        A_B_A_isolation=True,
        nonfinite_failure_control=True,
        model_weights_loaded=False,
        trials=results,
        calls_per_request=127,
        warmup_requests=2,
        timed_requests=6,
        capture_inclusive_ratio=ratio,
        required_ratio=0.95,
        activation_resources=resources,
        enclosing_graph_rng_lifetime_control=lifetime,
        passed=ratio < 0.95,
    )
