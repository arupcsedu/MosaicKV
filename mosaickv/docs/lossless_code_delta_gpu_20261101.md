# Consumed-input lossless GPU code-chain diagnostic

This separately authorized engineering stage follows the positive CPU-only
feasibility study. It does not alter that study's disabled GPU authorization.
Use only the same Description and MM-Vet inputs; no fresh sample selection,
calibration, powered confirmation, or paper campaign is authorized.

The frozen source-precision and four baseline precision adaptations retain
their exact minima, FP32 scales, quantizer codes, protected K/V, transport
corrections, selected positions, original logical positions, and generated
history. They are NOT the exact-selection algorithm or unmodified official
baselines. Offer the same code-byte storage option to all five methods. None
of the six original algorithm mechanisms is silently reactivated.

Adjacent layers' existing packed code bytes use 128-byte raw or sparse-XOR
blocks. This is a serialization block, not a selection-token block. Charge
four-byte offsets, masks, literals, and one request-private replay buffer per
stream role. Keep the original representation when those costs remove the
saving. GPU encoding transfers only its scalar allocation length to CPU.
There is no tensor offload, added approximation, changed scale or new fitting.

CPU tests precede GPU execution from a clean immutable SHA. GPU synthetic
controls must match CPU payloads, offsets and in-place/out-of-place replay.
Reconstruct each original K/V tensor bitwise. Every method/input must match
its frozen 128-token full-vocabulary logits and complete teacher-forced logits,
NLL, target mask and token count. Fresh request bindings and A-B-A CPU tests
prevent cross-request replay cursor or buffer sharing.

Two workers, each one A100 and 12 CPUs, have a 90-minute bound; aggregate use
must remain at most six A100s. Run only after the currently registered common
baseline timing chain completes. Use afterany CPU analysis so failed and
timed-out producers remain visible. Preserve every failure and raw record.

If net source storage improves, use five warmups and twenty randomized paired
128-token decode repetitions against its own unchanged representation. Record
CUDA events and synchronized wall time, p50/p95 and paired bootstrap intervals.
These are engineering-trial intervals, not independent quality evidence or
end-to-end latency. Encoding/proof cost is reported separately. This does NOT
authorize a latency claim against independently timed historical baselines.

Initial/final/decode-owned active bytes include actual backing storage and
replay scratch. Old control packets coexist for verification and are disclosed;
this is not a reduced whole-request allocated-memory experiment. Conversion
occurs AFTER the unchanged streaming prefill, so no prefill-peak reduction is
claimed. Common source-slot capacity is not equal encoded physical bytes.
PPL and FullKV-reference ROUGE-L remain separate, unchanged development metrics.
No result on these two consumed inputs establishes benchmark superiority.

## Separate constructor-lifetime follow-up

The `prepare_lifetimes` mode is a separately versioned, consumed-input diagnostic
for the same five methods. It releases decoded source tensors after both native
projections, releases redundant noncontiguous projection aliases after creating
their original contiguous reference, and releases unpatched trial K/V tensors
after sparse corrections have been copied. The common text constructor releases
its redundant trial pair too. No operation, correction or independent check is
removed; the frozen payload and all logits must remain bitwise identical.
The common quantizer also releases its FP32 preparation tensor and expanded
protection masks before entering the independent packing stage, and releases
the full FP32 quantizer input before integer serialization. This is a lifetime
change only: the same arithmetic, extrema, scale, rounding and bytes remain.

This follow-up enables the same prefill tensor-result observer for every method
and retains parent/new observations. It does not repeat the twenty paired decode
trials, because none of the decode math changes. An observed encoder/cache peak
is not whole-request GPU peak memory. Preserve the preceding code-delta result,
including failures; do not relabel constructor-lifetime evidence as an automatic
successful paper experiment. Submit only after the preceding GPU chain finishes.

## Accounting repair retry

Preserve the failed initial GPU chain: its primitive GPU controls passed, but
the independent end-of-decode accounting check detected duplicated resident
charges for the one shared replay buffer. `prepare_repair` binds that negative
decision and retries only these same two consumed inputs. Replay buffers belong
to `shared()` rather than every layer's `private()` residency; the independent
enumeration check remains unchanged. A CPU regression now constructs a real
cache, executes two K/V update epochs and checks cached-versus-enumerated storage
before and after updates. The retry includes the common lifetime optimization,
audited prefill storage, per-method partial construction records, explicit
parent baseline manifests, and the original five/twenty paired decode schedule.
Incomplete workers or missing timing records cannot pass its final analysis.
Primary generation and teacher-forcing measurements enable the same tensor
result auditing as the frozen parent. Only the separately labeled timed trials
disable that instrumentation. Do not compare an unaudited decode peak with the
parent's audited peak. The first repair's GPU admission was canceled while still
pending to enforce this measurement-boundary correction before any GPU inference.
