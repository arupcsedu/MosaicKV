# Bounded reader V28: complete observations without repeated history copying

This is a consumed-input storage/transport diagnostic. It changes measurement
implementation only, not model weights, selected tokens, cache values,
logical positions, generation, workspace, or numerical/memory thresholds.
It does not activate the six-mechanism algorithm or authorize a new cohort.

## Measured predecessor

V27 source `bd63bee79c4330bde79b16c62aefa22b993cc365` passed 128/128
predictions with bitwise-identical logits on both consumed inputs. Both source
workers also passed their complete registered published-KV lifecycle ledger.
The raw comparator timed out at its 15-minute execution limit, so V27 did
**not** establish paired whole-request byte matching. Its artifacts remain
unchanged at `/scratch/djy8hg/workdir/fox-review-v27/run-20261111-3`.

Every ATen constructor operation, including CPU-only selection work, invoked
`torch.cuda.memory_snapshot()`. The pinned implementation first serializes the
complete allocation history, then returns only `segments`. Each snapshot also
adds a history event. Repeatedly copying that growing prefix creates quadratic
instrumentation work. This is not evidence of slow model decoding.

## Exact reuse contract

The optional resolver is disabled by default. Only consecutive observations
of the same `prefill_layer_N_constructor_workspace` phase can reuse an actual
resolved physical-block map. It requires the same full GPU owner set, storage
identities, pointers, backing sizes, offsets, shapes, strides, dtypes and
continuously live weak base representatives. A dead owner followed by an
identical reused address cannot pass. No tensor/storage lifetime is extended.

Any owner change, missing continuity, phase change, publication, retirement,
decode update or growth boundary requires a fresh strict native snapshot.
No allocator rounding formula substitutes for measured physical blocks.
Returned evidence is copied so consumers cannot mutate the cached resolution.

Every original observation, synchronization and ownership delta remains.
Each delta records fresh-versus-reused provenance, the actual mapping hash,
source observation, owner/view fingerprint and continuity checks. Independent
CPU replay checks the contiguous origin chain and exact physical mapping;
missing proofs, lifecycle reuse and counter/hash changes fail closed.

The GPU operator canary additionally compares reused mappings against an
independent **fresh** strict allocator resolution at every tested step. It
tests aliasing, additions/removals, actual pointer reuse, resizing and lifecycle
barriers. Both source and raw arms use the same optional resolver policy.

## Unchanged acceptance and boundaries

Require complete 128-row finite logits and token parity, original logical
positions, zero fallback, all native allocation-generation `FREE_COMPLETED`
receipts and original-word source construction. Compare measured initial,
final and entire registered lifecycle peak for both GPU tensor backing and
physical allocator blocks. Constructor temporary outputs remain charged.

This ledger covers **published GPU KV and cache-required construction/reader
storage**, not total GPU memory or every ordinary model activation. CPU
position metadata is reported separately. The old 49 MiB lossless source target
remains unsupported. These two consumed inputs are not independent task-quality
confirmation, latency superiority or paper-scale evidence.

## Finite execution

Use `mosaickv/scripts/evicastkv_bounded_reader_v28.py launch --root NEW_ROOT`
from its clean immutable source with all six submodules initialized. A complete
CPU suite and GPU operator precede four one-A100/12-CPU/15-minute workers.
Aggregate A100 admission is at most six. CPU-only tests use four CPUs, 8 GiB
and 15 minutes on the authorized short-development `interactive` partition;
the paired analyzer uses 12 CPUs, 24 GiB and a 20-minute replay allowance.
Every stage uses write-once receipts and finite `afterany` continuation.
There is no continuous log polling, automatic fresh cohort or large campaign.
