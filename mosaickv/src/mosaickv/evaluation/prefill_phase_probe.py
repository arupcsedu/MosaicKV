"""Consumed-input phase labels for the existing tensor-storage observer.

Only Python strings, counters and byte totals are saved; never pin a tensor.
This probe does not change cache math or replace the original observer.
"""

from functools import wraps

from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache


class PhaseCache(RetiringPrefillCache):
    instances = None

    def __init__(self, *args, **kwargs):
        self.phase_stack, self.peak_events, self.previous_peaks = [], [], {}
        super().__init__(*args, **kwargs)
        if self.instances is None or self.instances:
            raise RuntimeError("one isolated profiling request required")
        self.instances.append(self)

    def observe(self, extra=()):
        super().observe(extra)
        for device, peak in self.peak_device_bytes.items():
            if peak > self.previous_peaks.get(device, 0):
                self.peak_events.append(
                    dict(
                        layer=self.next_layer,
                        device=device,
                        bytes=peak,
                        phase=tuple(self.phase_stack) or ("native_or_retirement",),
                    )
                )
                self.previous_peaks[device] = peak


def wrap_phase(name, function, instances):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if len(instances) != 1:
            raise RuntimeError("one active phase cache required")
        cache = instances[0]
        cache.phase_stack.append(name)
        try:
            return function(*args, **kwargs)
        finally:
            cache.phase_stack.pop()

    return wrapped
