"""Real lossless byte-graph controls, not model-quality observations."""

from dataclasses import replace

import torch


@torch.inference_mode()
def run():
    from mosaickv.evaluation import layer_code_delta as runtime
    from mosaickv.evaluation.precision_tier import PrecisionFormat, QuantizedBody
    from mosaickv.evaluation.replay_descriptor_policy import use

    originals = []
    for i in range(32):
        codes = torch.zeros(1024, device="cuda", dtype=torch.uint8)
        codes[i] = i + 1
        body = QuantizedBody(
            (1, 1, 8, 128),
            8,
            3,
            128,
            codes,
            torch.zeros((1, 1, 8, 1), device="cuda"),
            torch.ones((1, 1, 8, 1), device="cuda"),
        )
        originals.append(
            PrecisionFormat(
                body.shape,
                (body, replace(body, codes=codes.clone())),
                torch.empty(0, device="cuda", dtype=torch.int32),
                torch.empty(0, device="cuda", dtype=torch.int32),
                tuple(
                    torch.empty((1, 1, 0, 128), device="cuda", dtype=torch.bfloat16)
                    for _ in range(2)
                ),
            )
        )
    encoded, report = runtime.convert(originals)
    if not report["enabled"]:
        raise RuntimeError("actual compressible CUDA code fixture required")
    with use(True):
        ordinary, captured = runtime.bind(encoded), runtime.bind(encoded)
    reference, actual = ordinary[0].replay, captured[0].replay
    if any(reference.scratch[r].data_ptr() == actual.scratch[r].data_ptr() for r in actual.scratch):
        raise RuntimeError("captured decoder shares mutable request scratch")
    actual.prepare_graph()
    graphs, replays = {}, 0
    stream = torch.cuda.Stream()
    try:
        for _ in range(3):
            for i in range(32):
                expected = reference.view(i)
                result = actual.advance_for_graph(i)
                if actual.graph_bindings[i]:
                    graph = graphs.get(i)
                    if graph is None:
                        # The reference has already compiled the same decoder.
                        # No in-place warm decode may precede this capture.
                        graph = torch.cuda.CUDAGraph()
                        current = torch.cuda.current_stream()
                        stream.wait_stream(current)
                        with torch.cuda.stream(stream):
                            graph.capture_begin(capture_error_mode="thread_local")
                            try:
                                actual.decode_for_graph(i)
                            finally:
                                graph.capture_end()
                        current.wait_stream(stream)
                        graphs[i] = graph
                    graph.replay()
                    replays += 1
                for a, b, c in zip(result.pair, expected.pair, originals[i].pair, strict=True):
                    if not torch.equal(a.codes, b.codes) or not torch.equal(a.codes, c.codes):
                        raise RuntimeError("captured XOR replay changed an original code byte")
        if len(graphs) != 31 or replays != 93:
            raise RuntimeError("all nonraw layer frames must genuinely decode on each round")
        captures = len(graphs)
    finally:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for graph in graphs.values():
            graph.reset()
        graphs.clear()
        if "graph" in locals():
            del graph
        torch.cuda.synchronize()
        released = before - torch.cuda.memory_allocated()
        if released != 1024:
            raise RuntimeError("captured byte control did not retire its exact generator owners")
    return dict(
        bitwise_all_original_code_bytes=True,
        private_scratch_disjoint=True,
        first_replay_no_double_XOR=True,
        captures=captures,
        replays=replays,
        generator_metadata_retired_bytes=released,
        synthetic_fixture=True,
        independent_quality_N=0,
    )
