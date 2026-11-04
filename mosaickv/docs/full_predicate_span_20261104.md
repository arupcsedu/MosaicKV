# Bounded larger ATen predicate spans: consumed-input execution diagnostic

The two preceding November 4 timing runs are complete strict negatives and
remain immutable. The source-validation revision reduced construction work but
did not meet equal-or-faster request latency. Its packet, parent-logit and
storage controls passed. No independent quality cohort is authorized here.

## Intervention

Keep the same `torch.isclose`, finiteness, protected-row and independent
verification operations, dtypes, rounding and tolerances. Change only the
number of elements handled by each call. For up to 4,194,304 scalar elements,
use a single bounded predicate span instead of repeated 524,288-element spans.
The resulting Boolean conjunction, ascending correction indices and correction
values must match exactly. Larger tensors explicitly retain the old bounded
chunk path. Defaults remain unchanged; the experiment enables an explicit flag
in the immutable plan for every applicable method.

This is **not** a fused numerical kernel or removal of validation. The pinned
[PyTorch implementation](https://github.com/pytorch/pytorch/blob/a1cb3cc05d46d198467bebbb6e8fba50a325d4e7/aten/src/ATen/native/TensorCompare.cpp#L327)
evaluates elementwise predicates followed by Boolean reduction; regrouping does
not replace its arithmetic. Preserve failure coverage including final elements,
NaNs and protected mismatches. Report actual effective spans in fit metadata.

## Workspace and accounting

The larger ATen calls can increase **temporary construction** workspace. The
4-MiB explicit Boolean-mask cap still holds; it is not a bound on every ATen
temporary. Report the span and a conservative 64-bytes-per-element auxiliary
tensor bound, explicitly labeled theoretical, **not measured**. Operator trials
also record actual CUDA allocated-peak deltas including output tensors, and
patch backing bytes. Every full request still records CUDA allocated/reserved
maxima. These measurements do not certify whole-request KV peak ownership.
No workspace persists into decoding; frozen initial/final/decode-owned peak KV
storage must remain identical. Never substitute the theoretical bound for a
measured-results field.

## Finite fail-closed chain

Clean source -> full CPU suite -> two one-A100 controls on the same consumed
inputs. Before any request executes, run a synthetic CUDA operator comparison
at the matching 600/604-token geometry. Five warmups and twenty randomized
paired repetitions must show identical patch words and adjusted upper operator
ratio <=0.85 (10,000 bootstrap draws, tail 0.05/8). Save the complete operator
record even on failure. The checkpoint is loaded, but no model request is run
before that gate; this is operator evidence, not answer-quality evidence.

Only then execute all five frozen parent controls, followed by the registered
request timing (five warmups, twenty paired randomized repetitions). The strict
adjusted upper request-latency ratio remains <=1.00 against every measured
control, including the fastest PrefixKV representation. A strict pass alone
may release the bounded all-five-method timing extension. Do not launch fresh
evaluation, calibration or paper-scale work. All six main algorithm mechanisms
remain disabled in this precision/storage variant; methods are local storage
adaptations, not official-code performance claims. Independent quality N=0.
