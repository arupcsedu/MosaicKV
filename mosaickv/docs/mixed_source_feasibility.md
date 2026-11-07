# Last bounded CPU source-allocation screen

Preserve the negative rank and spatial-prototype/residual attempts. Changing
an early layer's representation also disrupted adjacent-layer code compression;
the corrected codec-coupled chooser retained the parent at all layers.

This separate screen tests whether source-channel anisotropy offers a feasible
replacement: use fixed groups of eight original features and 1/2/3/4-bit formats.
One/two-bit groups use the already tested four-iteration BF16 centroid table;
three/four-bit groups use the existing affine range quantizer. The original
three-bit choice remains available everywhere. No benchmark annotations,
generated answers, model weights, gradients or parameter updates are accessed.

Freeze the eight-feature grouping, all four formats, twenty-second per-layer
LP bound, and at most 128 monotone precision upgrades before execution. The LP
is a **fractional optimistic source-payload bound**, not measured cache storage.
Round to a deterministic integral layout and restore per-row feasibility using
bounded monotone precision upgrades, otherwise roll back completely to the
parent layout. Every actual non-exact image row must preserve the old source
MSE (+1e-12); no model-level KL/logit/token thresholds change.

Charge actual packed codes, 128-byte code-stream alignment padding, BF16 or
FP32 quantizer headers, per-group bit widths, two complete int32 offset arrays
and the format tag. Serialize each group of eight channels independently;
all row/feature coordinates remain original. Compare both raw CPU payload and
the common adjacent-layer code codec including replay workspace. Globally
choose formats using the tested codec-coupled dynamic program. Baseline totals
must reproduce the earlier captures exactly. Require >=10% net CPU source
payload reduction on **both** consumed inputs before considering a native
format implementation. Preserve LP timeouts, failed rounds and rejected layers.

These remain CPU-array feasibility measurements. The LP can identify a
precision allocation but cannot predict attention/logit/answer preservation.
Neither mixed-bit decoding nor source-level prototype/repair is supported by
the production reader. Source-fit time, real GPU cache ownership, whole-request
peak, all-six component utility and answer quality remain unvalidated. No GPU,
fresh-cohort or paper campaign follows automatically from this screen. If this
also fails, stop source-format expansion rather than increasing search/precision
grids or weakening numerical/model gates.
