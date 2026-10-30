# Encoded source -> K/V without a decoded-source allocation

This is distinct from the negative row-predictor study, which is stopped.
No selection, codec payload, source bits, values, logical positions, model
weights, precision or quality thresholds change. Both frozen selectors receive
the same execution candidate; no fresh cohort or paper campaign follows.

The validated dual kernel first allocates a decoded normalized-source matrix.
The new kernel reads the existing exact BF16 byte/exponent packet inside its
64-wide GEMM reduction tiles. An int16 count checkpoint every 128 values makes
literal addresses available without rescanning entire codec blocks. A second
64-value-half count preserves the existing GEMM reduction width/order. The
checkpoint cost is 16 bytes per 1024 values and is fully charged. Three
independent original-K/V bit checks and exact corrective patches remain.

The GPU microcontrol decodes all 65,536 BF16 patterns, including NaN payloads.
Then retention-one and complete 32-token own-output controls run on the same
two consumed inputs. Initial storage may increase by at most two percent and
EviCastKV must still fit 62 MiB initial and 96 MiB observed decode-owned peak.
No tensor-view accounting or hidden decoded-source cache is allowed.

Only complete numerical controls advance to five warmups and twenty paired
trials. Engineering eligibility requires reduced observed cache-owned peak
and a paired latency-ratio upper confidence bound <=1.05 against the validated
dual path, on each input/selector. A slower result remains an explicitly
reported memory–latency trade-off, not a default or a superiority claim.
The comparison is between exact storage execution policies, not byte-matched
selection algorithms. Whole-request HBM peak remains uncertified.
