"""Checkpoint-free native history projection feasibility, not an online reader.

The observed encoded-history projector is a large common kernel cost. Test a
lossless source materialization plus zero-workspace FP32 native GEMM instead.
Any different output word needs an explicitly counted original-bit repair.
"""

import random
import statistics
from types import SimpleNamespace

import torch
import triton

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

ROWS = (16, 32, 48, 64, 80, 96, 112)


class RepairBudgetError(RuntimeError):
    def __init__(self, diagnostic):
        super().__init__("native history correction storage exceeds unchanged 32-KiB epoch cap")
        self.diagnostic = diagnostic


def validate_rows(n):
    if type(n) is not int or n not in ROWS:
        raise ValueError("only complete 16-token history epochs through 112 supported")


def canonical_device(device):
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def workspace_buffers(device, buffers):
    device = canonical_device(device)
    if buffers is None:
        source = torch.empty((128, 4096), device=device, dtype=torch.bfloat16)
        return source, torch.empty_like(source)
    if (
        type(buffers) is not tuple
        or len(buffers) != 2
        or any(
            t.shape not in ((128, 4096), (192, 4096))
            or t.dtype != torch.bfloat16
            or t.device != device
            or not t.is_contiguous()
            for t in buffers
        )
        or buffers[0].untyped_storage().data_ptr() == buffers[1].untyped_storage().data_ptr()
        or buffers[0].shape != buffers[1].shape
    ):
        raise ValueError("two distinct original 128/192-row BF16 workspaces required")
    # Views allocate nothing; backing_bytes still charges the complete original
    # owners. The history operator's registered row set remains <=112.
    return buffers if buffers[0].shape[0] == 128 else tuple(t[:128] for t in buffers)


class Engine:
    def __init__(self, device, *, frozen=False, buffers=None):
        from mosaickv.evaluation.zero_workspace_lt import Plans

        if type(frozen) is not bool:
            raise ValueError("explicit frozen native history algorithm boolean required")
        if frozen:
            from mosaickv.evaluation.joint_all_layer_compatibility import FROZEN
            from mosaickv.evaluation.joint_kernel_compatibility import KernelPlans, fingerprint

            self.plans = KernelPlans(ROWS)
            options = [
                (algo, record)
                for algo, record in self.plans.candidates
                if fingerprint(record) == FROZEN
            ]
            if len(options) != 1:
                self.plans.close()
                raise NotImplementedError(
                    "previously audited native kernel unavailable at history rows"
                )
            for n, (a, b, d, _, _) in self.plans.plans.items():
                self.plans.plans[n] = a, b, d, *options[0]
        else:
            self.plans = Plans(ROWS, 4096)
        self.borrowed = buffers is not None
        try:
            self.source, self.output = workspace_buffers(device, buffers)
        except BaseException:
            self.plans.close()
            raise

    def project(self, packet, attention, component, target, repairs=None):
        from mosaickv.evaluation.dual_projection_kernel import apply_patch
        from mosaickv.evaluation.encoded_projection_kernel import _materialize, pointers
        from mosaickv.evaluation.shared_source_kernel import rotate_native

        n = packet.shape[-2]
        validate_rows(n)
        if (
            component not in (0, 1)
            or packet.archive.source.shape != (1, 1, n, 4096)
            or packet.pair
            or packet.slots is not None
            or packet.packed_literal_counts
            or target.shape != (1, 32, n, 128)
            or target.stride(-1) != 1
            or target.stride(-2) != 128
            or target.dtype != torch.bfloat16
        ):
            raise ValueError("complete identity history and original strided BF16 target required")
        body = packet.archive.source
        compiled = _materialize[(triton.cdiv(n, 16), 64)](
            *pointers(body),
            packet.literal_checkpoints,
            self.source.view(torch.uint16),
            n,
            WIDTH=4096,
            RAW=body.raw,
            PACK_COUNTS=False,
            num_warps=4,
        )
        if compiled.metadata.global_scratch_size or compiled.n_spills:
            raise RuntimeError("lossless history unpack introduced scratch or spills")
        self.plans.execute(
            self.source, (attention.k_proj, attention.v_proj)[component].weight, self.output, n
        )
        target[0].copy_(self.output[:n].view(n, 32, 128).transpose(0, 1))
        if component == 0:
            rotate_native(target, packet.archive.cos, packet.archive.sin, 0, n)
        if repairs is not None:
            apply_patch(repairs[component], target)

    def close(self):
        torch.cuda.synchronize()
        self.plans.close()


def make_packet(x, cos, sin):
    from mosaickv.evaluation.encoded_projection import EncodedFormat
    from mosaickv.evaluation.encoded_projection_kernel import checkpoints
    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.projection_archive import ProjectionArchive

    n = x.shape[2]
    validate_rows(n)
    if any(
        t.shape != (1, n, 128)
        or t.dtype != torch.bfloat16
        or t.device != x.device
        or not torch.equal(t[..., :64].view(torch.int16), t[..., 64:].view(torch.int16))
        for t in (cos, sin)
    ):
        raise ValueError("actual Llama duplicate-half BF16 rotary phases required")
    union = torch.arange(n, device=x.device, dtype=torch.int32)
    slots = union[None].expand(32, -1)
    body = encode(x)
    archive = ProjectionArchive(body, union, slots, n, cos, sin)
    return EncodedFormat(
        "encoded_projection",
        (1, 32, n, 128),
        archive=archive,
        output_slots=slots,
        literal_checkpoints=checkpoints(body),
    )


def duplicated_phases(theta):
    if theta.ndim != 3 or theta.shape[0] != 1 or theta.shape[-1] != 64:
        raise ValueError("one original 64-frequency Llama rotary table required")
    phases = torch.cat((theta, theta), -1)
    return phases.cos().bfloat16(), phases.sin().bfloat16()


@torch.inference_mode()
def study(*, frozen=False):
    from mosaickv.evaluation.dual_projection_kernel import apply_patch
    from mosaickv.evaluation.encoded_projection_kernel import project
    from mosaickv.evaluation.exact_storage import SparseBitPatch, decoded
    from mosaickv.evaluation.reconstruction_policy import use

    torch.manual_seed(20261105)
    torch.use_deterministic_algorithms(True)
    attention = SimpleNamespace(
        head_dim=128,
        k_proj=torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(4096, 4096, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    engine, records = Engine(torch.device("cuda"), frozen=frozen), []
    try:
        for n in ROWS:
            source = torch.randn((1, 1, n, 4096), device="cuda", dtype=torch.bfloat16)
            cos, sin = duplicated_phases(torch.randn((1, n, 64), device="cuda"))
            packet = make_packet(
                source,
                cos,
                sin,
            )
            del source
            owners = tuple(
                torch.full((1, 32, n + 7, 128), -71, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            )
            targets = tuple(t[..., :n, :] for t in owners)
            with use("carry_literals"):
                project(
                    packet.archive.source,
                    packet.literal_checkpoints,
                    attention,
                    packet.archive,
                    packet.output_slots,
                    targets,
                    component=0,
                )
                golden = targets[0].contiguous().clone()
                project(
                    packet.archive.source,
                    packet.literal_checkpoints,
                    attention,
                    packet.archive,
                    packet.output_slots,
                    targets,
                    component=1,
                )
                reference = (golden, targets[1].contiguous().clone())
                for c in (0, 1):
                    engine.project(packet, attention, c, targets[c])
                expected_source = decoded(packet.archive.source).reshape(n, 4096)
                if not torch.equal(
                    engine.source[:n].view(torch.int16), expected_source.view(torch.int16)
                ):
                    raise RuntimeError("native source materialization changed an original word")
                del expected_source
                patches = tuple(
                    SparseBitPatch.build(a, b) for a, b in zip(reference, targets, strict=True)
                )
                if backing_bytes(tuple(t for p in patches for t in p.tensors)) > 32768:
                    raise RepairBudgetError(
                        dict(
                            rows=n,
                            repair_bytes=backing_bytes(
                                tuple(t for p in patches for t in p.tensors)
                            ),
                            changed_words_by_component=[p.indices.numel() for p in patches],
                            maximum_absolute_error_by_component=[
                                float((a.float() - b.float()).abs().max())
                                for a, b in zip(reference, targets, strict=True)
                            ],
                            native_algorithm=engine.plans.plans[n][4],
                            prior_completed_epochs=records,
                            registered_cap_bytes=32768,
                        )
                    )
                for c in (0, 1):
                    apply_patch(patches[c], targets[c])
                if any(
                    not torch.equal(a.view(torch.int16), b.view(torch.int16))
                    for a, b in zip(reference, targets, strict=True)
                ):
                    raise RuntimeError("counted native-history repairs failed exact words")
                if any(not bool((t[..., n:, :] == -71).all()) for t in owners):
                    raise RuntimeError("native history wrote outside its original slice")
                torch.cuda.synchronize()
                stream = torch.cuda.Stream()
                graphs = {}
                for name in ("original", "native"):

                    def call(name=name, packet=packet, targets=targets, patches=patches):
                        for c in (0, 1):
                            if name == "original":
                                project(
                                    packet.archive.source,
                                    packet.literal_checkpoints,
                                    attention,
                                    packet.archive,
                                    packet.output_slots,
                                    targets,
                                    component=c,
                                )
                            else:
                                engine.project(packet, attention, c, targets[c], patches)

                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            call()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            call()
                    graphs[name] = graph
                torch.cuda.synchronize()
                trials, rng = [], random.Random(20261105)
                for trial in range(-5, 20):
                    order = ["original", "native"]
                    rng.shuffle(order)
                    for position, name in enumerate(order):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        torch.cuda.synchronize()
                        start.record()
                        graphs[name].replay()
                        end.record()
                        torch.cuda.synchronize()
                        trials.append(
                            dict(
                                trial=trial,
                                order=position,
                                arm=name,
                                warmup=trial < 0,
                                cuda_ms=start.elapsed_time(end),
                            )
                        )
                # Replay each graph again and verify every output, not only the
                # eager construction. Baseline gets the identical target shape.
                for name, graph in graphs.items():
                    graph.replay()
                    torch.cuda.synchronize()
                    if any(
                        not torch.equal(a.view(torch.int16), b.view(torch.int16))
                        for a, b in zip(reference, targets, strict=True)
                    ):
                        raise RuntimeError(f"captured {name} changed a repaired output word")
                means = {
                    name: statistics.mean(
                        r["cuda_ms"] for r in trials if r["arm"] == name and not r["warmup"]
                    )
                    for name in graphs
                }
                records.append(
                    dict(
                        rows=n,
                        encoded_source_bytes=backing_bytes(packet.tensors()),
                        explicit_workspace_bytes=backing_bytes((engine.source, engine.output)),
                        native_added_repair_bytes=backing_bytes(
                            tuple(t for p in patches for t in p.tensors)
                        ),
                        original_words_and_tail_sentinels_preserved=True,
                        means=means,
                        native_to_original_ratio=means["native"] / means["original"],
                        trials=trials,
                    )
                )
                del graphs
        return dict(
            complete=True,
            records=records,
            independent_quality_N=0,
            no_runtime_integration=True,
            no_model_or_dataset_access=True,
            numerical_repairs_are_real_counted_storage=True,
            operator_intervals_NOT_full_request_latency=True,
            all_epochs_operator_faster=all(r["native_to_original_ratio"] < 1 for r in records),
            real_model_parity_and_whole_request_gates_still_required=True,
        )
    finally:
        engine.close()
