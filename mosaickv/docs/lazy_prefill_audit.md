# Do not construct unused audit exclusions in timed requests

The updated construction profile attributes 85 ms cumulatively to 51,531
backing-storage identity calls. The retirement hook enumerated every prior
packet owner and every current attention parameter before checking whether
the per-operation audit was enabled. Timed requests explicitly disable that
audit; its fully checked excluded warmup remains enabled.

Create this exclusion set only inside the existing `if self.audit` branch.
Retain the full per-operation audit when enabled, independent backing-storage
enumeration, original-cache anti-alias checks, shape checks, final validation,
all original K/V words, all lifetimes and accounting. No tensor arithmetic,
allocation, cache policy or numerical threshold changes. Apply identically to
all methods. CPU tests prove the inactive path does not enumerate audit-only
weights while active auditing still does, and both preserve exact private K/V.
The measured 85 ms is instrumented attribution, not a promised request gain.
