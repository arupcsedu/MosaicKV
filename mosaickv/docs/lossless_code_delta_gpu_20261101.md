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
