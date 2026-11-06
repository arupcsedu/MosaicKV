"""Physical initial-cache cap, checked after binding and before any decode.

This is enforcement, not a capacity allocator. Passing a cap does not certify
largest-feasible filling, equal final/peak memory, quality, or paper admission.
"""

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class InitialByteBudgetExceeded(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(
            "initial physical cache exceeds cap before decode: "
            f"{evidence['actual_bytes']} > {evidence['cap_bytes']}"
        )


def validate_cap(cap):
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("initial physical byte cap must be a positive integer or None")


def inspect_initial_budget(cache, cap):
    validate_cap(cap)
    if cap is None:
        raise ValueError("explicit physical cap required for budget inspection")
    tensors = (
        *cache.shared_tensors,
        *(tensor for layer in cache.layers for tensor in layer.owned_tensors()),
    )
    actual = backing_bytes(tensors)
    if actual != cache.initial_bytes:
        raise RuntimeError(
            "bound cache initial counter differs from independent backing enumeration"
        )
    return dict(
        schema_version=1,
        boundary="bound_initial_unique_cache_backing_before_decode",
        cap_bytes=cap,
        actual_bytes=actual,
        within_cap=actual <= cap,
        shortfall_bytes=max(0, cap - actual),
        excess_bytes=max(0, actual - cap),
        before_first_decode=True,
        no_fullkv_fallback=True,
        includes_shared_and_bound_delta_code_allocations=True,
        largest_feasible_allocation_certified=False,
        not_a_whole_request_peak_cap=True,
    )
