"""Private corrected 192-row Source reader; no selection or weight changes."""

import time

import torch

from mosaickv.evaluation.joint_incoming_projection import Engine as OriginalEngine
from mosaickv.evaluation.joint_incoming_projection import _scope
from mosaickv.evaluation.source_192_feasibility import CAPACITY, Plans192, decode, ranges


class Engine192(OriginalEngine):
    def __init__(self, packet):
        state = _scope.get()
        if (
            state is None
            or state["policy"] != "kv"
            or state.get("source_rows", 128) != CAPACITY
            or state.get("frozen_plan") is not True
        ):
            raise ValueError("explicit private frozen joint-KV source-192 scope required")
        started = time.perf_counter()
        _, heads, n, dim = packet.shape
        ranges(n)
        if heads != 32 or dim != 128:
            raise ValueError("pinned full-MHA Llama source geometry required")
        self.plans = Plans192(n)
        try:
            self.source = torch.empty(
                (CAPACITY, 4096), device=packet.cos.device, dtype=torch.bfloat16
            )
            self.output = torch.empty_like(self.source)
            self.incoming = torch.zeros(4096, device=packet.cos.device, dtype=torch.bfloat16)
        except BaseException:
            self.plans.close()
            raise
        self.extra_tensors = (self.incoming,)
        self.tail, self.chunk_rows = n % CAPACITY, CAPACITY
        self.closed, self.fold_values = False, True
        self.project_calls = self.matmul_calls = self.joint_calls = 0
        self.construction_wall_seconds = time.perf_counter() - started
        state["frozen_plan_engines"] += 1

    def check_source(self, packet, reference):
        for first, take in ranges(packet.shape[2]):
            self.source.view(torch.int16).fill_(32767)
            decode(packet, self.source, first, take, CAPACITY)
            if not torch.equal(
                self.source[:take].view(torch.int16),
                reference[0, first : first + take].view(torch.int16),
            ):
                raise RuntimeError("source-192 decoder changed an original source word")

    def project(self, packet, attention, component, target, *, inspect_kernel=False):
        from mosaickv.evaluation.repaired_source_tile import validate
        from mosaickv.evaluation.shared_source_kernel import rotate_native

        if self.closed:
            raise RuntimeError("closed private source-192 engine")
        validate(packet, attention, component, target)
        _, heads, n, dim = packet.shape
        if n % CAPACITY != self.tail:
            raise ValueError("request source geometry changed")
        weight = (attention.k_proj, attention.v_proj)[component].weight
        for first, take in ranges(n):
            tail = first + take == n
            decode(packet, self.source, first, take, CAPACITY, self.incoming if tail else None)
            self.plans.execute(self.source, weight, self.output, take + int(tail))
            target[0, :, first : first + take, :].copy_(
                self.output[:take].view(take, heads, dim).transpose(0, 1)
            )
            self.matmul_calls += 1
            self.joint_calls += int(tail)
        if component == 0:
            rotate_native(target, packet.cos, packet.sin, 0, n)
        self.project_calls += 1
        if inspect_kernel:
            return dict(
                algorithm_workspace_bytes=0,
                explicit_workspace_bytes=self.workspace_bytes,
                incoming_words_verified_by_full_parent_logit_gate=True,
            )

    def record(self):
        return dict(
            super().record(),
            source_chunk_rows=CAPACITY,
            two_distinct_128_row_BF16_owners=False,
            two_distinct_192_row_BF16_owners=True,
            extra_source_output_bytes_vs_128=2**20,
            numerical_corrections_must_be_fitted_and_charged=True,
        )
