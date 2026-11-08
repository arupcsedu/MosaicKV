"""CPU proof for independent modulo-eight fields in original 24-bit words.

Guard bits prevent carries/borrows from leaking into neighboring three-bit
quantizer codes. No quantization, floating-point arithmetic or model fitting.
These oracles do not activate a new serving path.
"""

import numpy as np

LOW = np.uint32(0x6DB6DB)  # Two low bits in each of eight fields.
HIGH = np.uint32(0x924924)  # One carry/borrow guard per field.
FULL = np.uint32(0xFFFFFF)


def validate(a, b):
    if (
        not isinstance(a, np.ndarray)
        or not isinstance(b, np.ndarray)
        or a.dtype != np.uint32
        or b.dtype != np.uint32
        or not a.size
        or a.shape != b.shape
        or np.any(a > FULL)
        or np.any(b > FULL)
    ):
        raise ValueError("matching nonempty original 24-bit uint32 source words required")


def add(a, b):
    validate(a, b)
    return (((a & LOW) + (b & LOW)) ^ ((a ^ b) & HIGH)) & FULL


def subtract(a, b):
    validate(a, b)
    guarded = (a | HIGH) - (b & LOW)
    return (guarded & LOW) | ((a ^ b ^ ~guarded) & HIGH)
