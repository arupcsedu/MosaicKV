"""Private single-consumption reuse of an already computed guarded readout."""

from contextlib import ExitStack
from unittest.mock import patch

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class ReadoutReuse:
    def __init__(self, norm, head):
        self.norm, self.head = norm, head
        self.pending_norm = self.pending_logits = None
        self.staged = self.norm_reuses = self.head_reuses = self.peak_activation_bytes = 0
        self.borrowed_stages = 0
        self.stack = ExitStack()

    def stage(self, passed, parent_norm, active_norm, parent_logits, active_logits):
        if self.pending_norm is not None or self.pending_logits is not None:
            raise RuntimeError("previous guarded readout was not consumed")
        if (
            passed.numel() != 1
            or passed.dtype != torch.bool
            or parent_norm.shape != active_norm.shape
            or parent_norm.shape != (1, 1, 4096)
            or parent_logits.shape != active_logits.shape
            or parent_logits.ndim != 3
            or parent_logits.shape[:2] != (1, 1)
            or any(
                t.device != parent_norm.device or t.dtype != parent_norm.dtype
                for t in (active_norm, parent_logits, active_logits)
            )
            or passed.device != parent_norm.device
        ):
            raise ValueError("complete native same-device/dtype guarded readout required")
        self.pending_norm = torch.where(passed, active_norm, parent_norm)
        self.pending_logits = torch.where(passed, active_logits, parent_logits)
        self.peak_activation_bytes = max(
            self.peak_activation_bytes, backing_bytes((self.pending_norm, self.pending_logits))
        )
        self.staged += 1

    def stage_borrowed(self, normalized, logits):
        """Borrow already selected native graph outputs until the next outer readout.

        The producer may retire its graph, but must not overwrite these tensors
        before consumption. Strong tensor references retain their true backing.
        No clone or hidden new ownership is introduced here.
        """
        if self.pending_norm is not None or self.pending_logits is not None:
            raise RuntimeError("previous guarded readout was not consumed")
        if (
            normalized.shape != (1, 1, 4096)
            or logits.ndim != 3
            or logits.shape[:2] != (1, 1)
            or normalized.device != logits.device
            or normalized.dtype != logits.dtype
        ):
            raise ValueError("complete aligned native selected readout required")
        self.pending_norm, self.pending_logits = normalized, logits
        self.peak_activation_bytes = max(
            self.peak_activation_bytes, backing_bytes((normalized, logits))
        )
        self.staged += 1
        self.borrowed_stages += 1

    def __enter__(self):
        original_norm, original_head = self.norm.forward, self.head.forward

        def norm(hidden):
            if self.pending_norm is None:
                if self.pending_logits is not None:
                    raise RuntimeError("guarded normalization consumed more than once")
                return original_norm(hidden)
            if (
                hidden.shape != self.pending_norm.shape
                or hidden.dtype != self.pending_norm.dtype
                or hidden.device != self.pending_norm.device
            ):
                raise ValueError("outer final normalization geometry changed")
            result, self.pending_norm = self.pending_norm, None
            self.norm_reuses += 1
            return result

        def head(hidden):
            if self.pending_logits is None:
                return original_head(hidden)
            if self.pending_norm is not None:
                raise RuntimeError("guarded normalization must precede vocabulary readout")
            if (
                hidden.shape != (1, 1, 4096)
                or hidden.device != self.pending_logits.device
                or hidden.dtype != self.pending_logits.dtype
            ):
                raise ValueError("outer final vocabulary geometry changed")
            result, self.pending_logits = self.pending_logits, None
            self.head_reuses += 1
            return result

        self.stack.enter_context(patch.object(self.norm, "forward", norm))
        self.stack.enter_context(patch.object(self.head, "forward", head))
        return self

    def __exit__(self, *args):
        self.stack.close()
        complete = self.staged == self.norm_reuses == self.head_reuses
        self.pending_norm = self.pending_logits = None
        if args[0] is None and not complete:
            raise RuntimeError("guarded readout was not consumed exactly once")

    def report(self):
        return dict(
            staged=self.staged,
            borrowed_native_graph_stages=self.borrowed_stages,
            normalization_reuses=self.norm_reuses,
            vocabulary_reuses=self.head_reuses,
            parameters_changed=False,
            peak_temporary_activation_bytes=self.peak_activation_bytes,
            activation_storage_not_claimed_as_KV_savings=True,
        )
