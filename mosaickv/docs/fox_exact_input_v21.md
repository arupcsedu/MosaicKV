# V21: funded original-input repair, consumed inputs only

V20 established bitwise original-selected K/V transport and real initial/final/
decode-owned byte matching for the raw exact-selection controls. It did not
make the three-bit full-source representation lossless: changing to native
projection repaired some greedy errors but not all. Do not relabel that result.

This follow-up tests one causal intervention, not another precision/ratio sweep.
It uses the same two consumed inputs, frozen 49 MiB packet, native full-width
BF16 projections, original RoPE, original mandatory policy, and common raw
append-only history. Model weights are frozen. No task answers, reference
annotations, logits, or generated prefixes enter the constructor.

## Transaction

1. Materialize the native-quantized parent K/V.
2. Replace exact text K/V duplicated across every head with one original BF16
   normalized-input row per text position. Preserve the original full matrix
   shape and operation order, rather than projecting only a text-sized matrix.
   Retain nontext exact overlays. Verify every parent K/V bit; if needed,
   account for exact text-only numerical corrections rather than ignore them.
3. Compute the actual unique-owner bytes of the complete new representation.
   Only genuinely freed storage may fund additional original input rows.
4. Choose one fixed allocation ranked by prompt demand multiplied by normalized
   input quantization error. Store original BF16 rows and their position metadata;
   the three-bit source codes remain stored and charged. No assumptions about
   free source-code retirement or a guaranteed two-fold saving are made.
5. Verify all previously protected original K/V, exact logical positions,
   nonaliasing, immutable model weights, and the unchanged 49 MiB initial cap.

## GPU arms and gates

The native parent must first replay V20's saved 128 logits bitwise. The text-only
replacement must then reproduce that native parent bitwise for all 128 logits.
This representation-only control must preserve the parent's failures too.

The additional-row repair must make a nonempty change and pass both the
unchanged registered 32-prediction and full 128-prediction FullKV guards. The
bound remains KL <= 0.02, maximum logit difference <= 1.0, and greedy-token
agreement 1.0. All rows, failures, real storage, and diagnostic latency are saved.

The constructor must decline if full backing costs cannot fit; no hidden
FullKV fallback or arbitrary uncounted residual is allowed. These are initial
caps, not hard lifetime caps. Final and observed decode-owned peaks include raw
history and observed reconstruction workspace. Retained verifier prefills and
all compared packet families are disclosed separately. Exact whole-request
KV-only peak certification remains false.

Both comparison source formats have different representation sizes after the
transaction. They are **not** called byte matched. V20's separate common-
transport byte-matching proof remains unchanged. No superiority or benchmark-
quality claim follows from a consumed-input numerical diagnostic.

The finite launcher runs complete CPU tests, two one-A100/12-CPU workers with
30-minute limits, and `afterany` CPU verification. It does not authorize new
identities, calibration, fresh confirmation, or a large campaign. A failed
diagnostic is preserved and cannot be promoted by increasing sample count.

## CPU infrastructure retry

The first canonical V21 full suite on `udc-aw34-17c0` repeatedly entered
uninterruptible filesystem I/O wait while progressing without assertion
failures. Pytest's default file-descriptor capture repeatedly reads/truncates
temporary files on shared scratch. A narrowly scoped technical retry may use
`--cpu-capture-mode sys` to keep global Python output capture in memory; native
stdout still reaches the immutable outer CPU log. The exact command and capture
mode are recorded. Legacy callers retain default `fd`, the unchanged command,
and the old default 540-second timeout.

Original CPU job `22811390` timed out after its registered 1,200-second full-
suite allowance; no GPU was admitted. The retry launch binds that immutable
failure receipt. Its source SHA and negative artifacts remain unchanged.

V21's explicit full-suite timeout remains 1,200 seconds, with identical tests,
assertions, thread counts, AVX2 controls, model candidate, input identities,
storage accounting, and numerical guards. The original incomplete run must be
preserved. No GPU worker is admitted on a CPU timeout or missing receipt.
