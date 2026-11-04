# Atomic unchanged-predicate transport validation

Target only the measured construction bottleneck on the same two consumed
inputs. Do not alter source fitting, selected values/positions, predicates,
thresholds, sparse patches, reader policy or cache bytes. Default execution
remains immediate. Explicit deferred execution collects at most 256 scalar
Boolean results from finite checks, protected-row checks and the **same**
BF16 torch.isclose guard. It checks their conjunction at construction commit,
before exposing any compressed cache to decoding. Shape/dtype checks remain
immediate. An error aborts the request and clears all private state. First
failing check retains its ValueError/RuntimeError/AssertionError category;
deferred failure diagnostics identify the predicate, not a retained full tensor.

Only flags survive: no KV/reference tensors or benchmark labels are retained.
Record actual flag-tensor peak bytes and commit count. Clear flags before
decode; initial/final/decode-owned peak storage must remain identical. This
does not certify whole-request GPU peak or allocator overhead; separately
retain CUDA allocated/reserved maxima. This is scheduling of validation, not
weaker validation or a new compression algorithm.

Finite DAG: full CPU suite -> all-five exact packet/logit/storage controls ->
paired five-warmup/twenty-repeat timing -> strict analyzer. Apply the policy
identically to every compared method. Frozen packet hashes from the preceding
immediate-construction control must match, not just generated tokens. Strict
adjusted upper latency ratio <=1.00 versus **every** measured PrefixKV control
is required to launch the already registered all-five timing. <=1.05 is only a
trade-off. Preserve all prior negative artifacts. At most two workers, one
A100/12 CPUs each, <=6 A100 aggregate, <2h shards. No fresh cohort, retuning,
calibration or paper campaign. Independent quality N remains zero.
