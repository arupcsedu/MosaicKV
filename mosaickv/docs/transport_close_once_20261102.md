# Bounded numerical verification with one successful synchronization

This is a construction-only optimization, not a numerical threshold change or
an algorithm revision. The consumed nested phase study found approximately
0.17 seconds per request in the repeated assert_transport_close calls. Each
component previously synchronized once for each 524,288-scalar chunk.

The pinned PyTorch comparison implementation invokes torch.isclose(actual,
expected, rtol=0.01, atol=2**-6, equal_nan=False), followed by torch.all. On
compatible BF16 tensors the successor invokes that same predicate on every
same-sized chunk, reduces each on-device and synchronizes their conjunction
once. Only scalar flags survive across chunks. Do not upcast the predicate,
drop any scalar, remove finite/protected-position checks, change patches or
alter a tolerance. Metadata mismatches and failures retain upstream assertions.

CPU tests compare both predicates at BF16 boundaries, NaN, infinity, signed
zero, noncontiguous tensors, dtype mismatch and chunk boundaries. These tests
are not GPU evidence. Any GPU follow-up must preserve frozen packet hashes,
storage, prefill logits and the original independent full-decode evidence.
The constructor-memory boundary must be checked before claiming the prior
workspace peak still applies. No fresh data or paper campaign follows from
this change. Preserve the prior full-request timing result and only proceed
if its measured remaining overhead warrants this bounded diagnostic.

The finite afterany follow-up first requires the completed parent timing
analysis. If every registered timing comparison passes, it submits nothing.
Otherwise it runs the complete CPU suite, two one-A100 constructor workers,
and dependent CPU analysis. Each worker audits memory on an excluded first
warmup for every method, then checks all original packet hashes, prefill logits
and physical bytes on every profiled construction. Require lower observed
constructor CUDA peak than every baseline on both inputs. This remains the
encoder-overlap boundary, not whole-request GPU peak. Failed/incomplete gates
stop the chain; they never trigger a larger cohort or weakened thresholds.

Only after that gate passes, reuse the same-source CPU proof for two full
request timing workers with five warmups and twenty repetitions per method.
The existing dependent analyzer checks all 250 requests and reports paired
latency, preserving every negative result. No perpetual controller, repeated
log polling, fresh IDs, label-driven changes or paper-writing promotion.
