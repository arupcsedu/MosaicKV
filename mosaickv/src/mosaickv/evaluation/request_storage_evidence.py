"""Keep independently audited peak evidence separate from fast timing counters."""

ENDPOINTS = (
    ("initial_encoded_bytes", "initial_active_kv_storage_bytes"),
    ("final_encoded_bytes", "final_active_kv_storage_bytes"),
)


def _endpoints(measured, frozen):
    for name, original in ENDPOINTS:
        if (
            type(measured[name]) is not int
            or measured[name] <= 0
            or measured[name] != frozen[original]
        ):
            raise RuntimeError(
                f"changed physical endpoint {name}: {measured[name]} != {frozen[original]}"
            )


def audited(measured, frozen):
    _endpoints(measured, frozen)
    if (
        measured["tensor_result_auditing"] is not True
        or measured["peak_measurement_admissible"] is not True
    ):
        raise RuntimeError("physical peak gate requires independently observed tensor workspace")
    actual = measured["decode_owned_peak_bytes"]
    expected = frozen["peak_active_kv_storage_bytes"]
    if type(actual) is not int or actual != expected:
        raise RuntimeError(f"changed AUDITED physical peak: {actual} != {expected}")
    return {**{k: measured[k] for k, _ in ENDPOINTS}, "decode_owned_peak_bytes": actual}


def timed(measured, frozen, *, expected_counter):
    _endpoints(measured, frozen)
    if (
        measured["tensor_result_auditing"] is not False
        or measured["peak_measurement_admissible"] is not False
        or measured["decode_owned_peak_bytes"] is not None
    ):
        raise RuntimeError("timing counters cannot masquerade as audited physical peaks")
    counter = measured["reported_counter_peak_bytes"]
    if (
        type(counter) is not int
        or counter <= 0
        or type(expected_counter) is not int
        or counter != expected_counter
        or counter > frozen["peak_active_kv_storage_bytes"]
    ):
        raise RuntimeError(
            f"changed uninstrumented counter: {counter} != {expected_counter}, "
            f"or exceeds audited peak {frozen['peak_active_kv_storage_bytes']}"
        )
