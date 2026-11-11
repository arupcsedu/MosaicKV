# V22 instrumentation-only correction

The source-row candidate remains a negative consumed-input result. Neither
its selection policy, four-step horizon, loss, byte cap, protected K/V, nor
32/128-prediction guards change in this correction.

## Observed defect

Both real-model scorer calls saved 4,909 tensors. The original observer rescans
all surviving weak references on every pack and unpack, including references
to excluded weights. This is quadratic host bookkeeping. The recorded
forward/backward intervals were 48.52 and 46.48 seconds, while the FullKV draft
was about 0.19 seconds and source materialization about 0.065 seconds.

CPU controls confirmed quadratic owner-lookup counts and identical peaks,
logits, and signed scores under an incremental prototype. This does not by
itself establish real GPU speed or a quality improvement.

## Correction and verification

Count live weak references per unique `(device, pointer, backing bytes)` owner.
Skip weights before registration. Weak callbacks decrement owner counts without
retaining a tensor or observer; unpack no longer rescans unrelated tensors.
Explicit close drops bookkeeping. Storage mutation during a save is rejected.
Model mathematics and the original backing-storage accounting boundary remain
unchanged; neither allocator nor whole-request KV peak is certified.

Two one-A100 workers run the old and new four-step scorer on the same consumed
inputs, with counterbalanced order. Compare every signed score bitwise, both
native logit digests, saved-storage peaks, cache isolation, and model state.
Save the complete score vectors and four-step logits. Rebuild the corrected
packet and require its content hash to equal the old V22 packet before reusing
the old 128-prediction result. There is **no 128-token GPU rerun** or new cohort.

Any speed change is an instrumentation result, not an end-to-end superiority
claim. The old negative quality result stays negative even if this gate passes.

## Derived-report correction

The original V22 analyzer inherited the filename `GATE_exact_input_v21.json`.
Its contents and source are V22, but that misleading filename is preserved as
an immutable historical artifact. CPU analysis may separately regenerate
`analysis-v22-corrected/GATE_exact_input_v22.json` using the saved predictions,
recording distinct producer and analysis source SHAs. It changes no metrics,
thresholds, identities, or outcome. No GPU inference is needed for this repair.

Full-prompt source versus compact Top-K peak matching remains unsupported under
the existing full-materialization reader. Do not fabricate equality with dummy
baseline padding or omit required reconstruction workspace.
