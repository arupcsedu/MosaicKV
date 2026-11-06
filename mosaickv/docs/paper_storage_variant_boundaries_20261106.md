# Storage-variant admission and whole-request memory boundary

This update does not turn a storage codec into the original six-component
algorithm. The historical `mosaickv_exact` ID is preserved, but the current
measured candidate is **EviCastKV shared-source precision/storage variant**.
Forecasting, evidence-graph construction, value-aware utility, submodular
selection, prototype/residual tiers and uncertainty-guided repair are inactive.
No configuration flag, namespace or measurement can substitute for their
independent activation and correctness/quality evidence.

## What remains valid

The previous 256-image COCO predictions, references, failures, selective
recovery and corpus-CIDEr bootstrap remain immutable. The candidate-versus-
FullKV quality result is not a compressed-baseline superiority comparison.
Old missing KV peaks remain missing. Previous whole-request allocator values
remain tagged with their actual producer version.

The historical shared-byte contract in `shared_byte_comparison_v2.yaml`
describes a different, FP16 exact-selection candidate and different runtime.
Its results and policy fingerprints cannot be assigned to this BF16 codec
variant. Its stricter equality, provenance, feasibility and no-FullKV-latency
rules are retained as admission requirements, not silently bypassed.

## Whole-request CUDA KV envelope

`WholeRequestEnvelope` starts before media loading/processing, resets the CUDA
peak exactly once, suppresses legacy interior resets, and closes only after
detokenization, packet retirement and synchronization. It measures allocations
from every PyTorch thread and stream, including asynchronous construction.

Let W be unique CUDA backing of all model parameters and registered buffers,
verified to retain the same backing throughout the request. Let P be the
whole-request maximum PyTorch allocation. Every request-owned CUDA KV tensor
is part of this allocator, so:

    CUDA KV peak <= P - W.

This is a **conservative certified upper bound**, not an exact KV peak. It
also includes activations, exported GPU logits and other request allocations.
Do not subtract the entire pre-request allocation: it may contain unknown
retained buffers. Reject nonresident/mixed-device models, incomplete requests,
invalid counter orderings and changing model backing.

This certificate does not cover CPU/offloaded KV, vendor-internal non-tensor
workspace, total process HBM, or allocations outside the audited PyTorch CUDA
storage path. Allocated, reserved, observed decode-owned peaks and the envelope
are different fields. An envelope cannot prove exact cross-method peak equality.

## Byte-matching feasibility, without manufacturing equality

The bounded diagnostic reuses one already-consumed COCO input, the same pinned
LLaVA-1.5 checkpoint, processor, BF16 eager backend, prompt, media, seed and
128-token decode for FullKV and five frozen precision/storage adaptations.
The compressed controls are Prompt Top-K, PrefixKV, VL-Cache and LOOK-M local
ports with the previously frozen common precision/runtime optimizations. These
are **not official implementations or paper-native reproductions**.

The candidate's previously measured per-sample initial bytes are the proposed
cap. Existing frozen baseline capacities are measured to discover feasibility,
not claimed to enforce that cap. Rows above it or with unequal achieved bytes
are explicitly rejected. Do not add dummy padding, reinterpret a nominal ratio
as encoded bytes, or silently shrink one method after looking at its score.

`comparison_admission` preserves missing/failed predictions and rejects unequal
initial/final/audited decode peaks, missing complete-request evidence, fallback,
and mismatched controls. Even a feasibility pass still needs independently
bound largest-feasible allocation certificates from `shared_byte_contract`.
It never authorizes a paper claim from an allocator upper bound.

## Finite execution

The new clean-source chain is CPU suite -> six one-A100 feasibility workers ->
afterany CPU analysis. One GPU/12 CPUs/48 GiB/two hours per worker, at most six
A100s in aggregate via the shared scheduler controller. No new data is
materialized; no large or independent campaign launches automatically.

The decision records which blocker actually remains. If packet capacities do
not meet the shared physical budget, a separately versioned byte-driven codec
allocator is required. If exact whole-request KV peaks remain unknown, complete
allocation ownership/lifetime instrumentation is required. Neither issue can
be resolved by renaming the existing measurements.

## Physical cap enforcement

`initial_physical_byte_cap` is an optional runtime guard, disabled in historical
requests. It independently enumerates the bound cache **after** delta-code
binding and **before** the first decode, rather than checking serialized packet
size. Binding can add live owners, so packet bytes alone are insufficient.
On an exceeded cap, private graph/workspace owners are retired and a structured
`InitialByteBudgetExceeded.evidence` record preserves attempted bytes, cap,
excess, and the before-decode boundary. No FullKV fallback occurs.

Enforcement does not resize the baseline or certify its largest feasible
allocation. An over-budget frozen baseline is unsupported, not a smaller-cache
performance result. The dedicated `--verify-initial-cap` consumed-input mode
checks both an exactly fitting candidate and expected pre-decode baseline
rejection. A rejection is explicitly labeled correctness evidence, not a failed
benchmark score or successful quality prediction.

The first diagnostic's controller omitted `--apply`. Its dry-run CPU controller
was canceled before any worker started and replaced by an apply-enabled
controller without changing the GPU source or array. The launcher now includes
`--apply` and has a regression preventing silent dry-run admission.

## Certifying a peak inequality on the existing quality cohort

An exact KV peak is not necessary to prove a conservative peak inequality.
For each example, candidate CUDA KV peak is at most its complete-request
allocator envelope U, while FullKV peak is at least its measured final KV F.
Therefore the peak ratio is at most U/F, and peak reduction is at least 1-U/F.
If U < F this proves a strict reduction without calling U an exact KV peak.
It does not prove a reduction against compressed baselines, and cannot satisfy
an equal-measured-peak comparison by itself.

`--recertify-existing-memory` repeats only the frozen candidate's 256 existing
quality inputs, with the unchanged 128-token generation and label-blind codec.
The original 512 paired prediction records are SHA-256 bound in source. Every
new row must exactly match its previous token IDs and initial/final storage.
The new measurement covers complete requests, not just decoding. Previously
missing exact peaks remain null; upper bounds and proven inequalities are
separate new observations. This is memory recertification, not a new quality,
freshness, baseline-superiority or full-six-component experiment.

Sixteen one-GPU shards (16 existing inputs each) and a finite afterany analysis
reuse the existing ledger. No benchmark scoring is repeated and no parameters
are chosen from answer differences. Any changed prediction, endpoint, missing
row or failed producer blocks complete-cohort certification.
