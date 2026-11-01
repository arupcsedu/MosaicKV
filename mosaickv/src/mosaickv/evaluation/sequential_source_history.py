"""Common bit-exact source history with one eager K-then-V layer lease.

No history eviction, quantization or phase renumbering. Frozen validated chunk
construction and consolidation retain their three original-bit checks.
"""

import torch

from mosaickv.evaluation.sequential_kv_lease import SequentialLayer, fill_component
from mosaickv.evaluation.source_history import SourceHistoryLayer


class SequentialSourceHistoryLayer(SequentialLayer, SourceHistoryLayer):
    def materialize_component(self, component):
        if not self.leased or component not in (0, 1):
            raise RuntimeError("private single layer lease and KV component required")
        destination = self.leased[0]
        fill_component(self.representation, self.attention, component, destination)
        offset = self.prompt_length
        for packet in self.source_sealed:
            end = offset + packet.shape[-2]
            fill_component(packet, self.attention, component, destination[..., offset:end, :])
            offset = end
        destination[..., offset : offset + self.tail_used, :].copy_(
            self.history[component][..., : self.tail_used, :]
        )
        if offset + self.tail_used != self.get_seq_length():
            raise RuntimeError("sequential source history dropped or duplicated a position")
        self.current_component = component
        self.component_fills = getattr(self, "component_fills", 0) + 1


def paired_run(adapter, prefill, formats, *, audit_history=True, **kwargs):
    """Compare original raw history against exact source history, not FullKV quality."""
    import gc

    from mosaickv.evaluation.storage_runs import run

    if type(audit_history) is not bool or kwargs.pop("sealed", True) is not False:
        raise ValueError("explicit common raw comparator and history-audit policy required")
    if any(
        kwargs.get(k)
        for k in (
            "source_history",
            "sequential_lease",
            "sequential_source_history",
            "consolidated_history",
        )
    ):
        raise ValueError("nested history/lease policies are unsupported")
    kwargs.pop("audit", None)
    original, baseline = run(
        adapter,
        prefill,
        formats,
        sealed=False,
        sequential_lease=True,
        audit=audit_history,
        **kwargs,
    )
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    actual, details = run(
        adapter,
        prefill,
        formats,
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        source_history_constructor="direct_batched_checks",
        consolidated_history=True,
        audit=audit_history,
        **kwargs,
    )
    if not torch.equal(original, actual):
        raise RuntimeError("source history changed an original raw-history logit bit")
    if (
        baseline["storage"]["initial_active_kv_storage_bytes"]
        != details["storage"]["initial_active_kv_storage_bytes"]
    ):
        raise RuntimeError("history codec changed initial prompt storage")
    details["storage"]["raw_history_control"] = dict(
        logits_bitwise=True,
        storage=baseline["storage"],
        decode_wall_seconds=baseline["decode_wall_seconds"],
        decode_cuda_ms=baseline["decode_cuda_ms"],
        both_one_KV_lease=True,
        audit_enabled=audit_history,
    )
    details["timing_includes_tensor_audit_instrumentation"] = audit_history
    return actual, details
