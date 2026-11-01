# FP32-metadata, bit-preserving tiling diagnostic

This bounded same-consumed-input engineering experiment follows the negative
compact-header experiment. FP16 source headers changed the Description answer
despite passing the short-prefix guard; preserve that result. Do not use
compact headers or select precision by task. No fresh identities are authorized.

Retain the original FP32 minimum/scale, three-bit source codes, original phases,
exact protected rows, sparse numerical corrections and lossless text-source
representation. Change only the fused projection's output-head tiling, selecting
one global group from the previously registered 64-case FP32 operator study.
Promotion requires all components bitwise and at least 10% operator improvement.
Do not refit numerical corrections after changing tiling.

Run the old shadow, unchanged FP32 full-source construction, lossless text-source
parent, and tiled parent on both consumed inputs. Require identical constructed
payloads and model logits at 33 forced-prefix and 128 greedy positions between
the tiled parent and its untiled control. Preserve the unchanged 32-token FullKV
KL/logit/token gate, retention-one bypass, original-source retirement, immutable
storage and replay controls. Record initial/final/decode-owned peak bytes,
CUDA and wall decode time; construction and whole-request GPU peaks are not
certified. No training, benchmark-label fitting, prototype or repair is enabled.

The two inputs are development-only, not paper evidence. A speed improvement
cannot establish equal-quality smaller-cache superiority against the strongest
baseline. Do not launch calibration or larger campaigns from this diagnostic.
