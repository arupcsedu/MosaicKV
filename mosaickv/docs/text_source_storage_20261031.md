# Common, bit-exact text-source storage diagnostic

Use only the same consumed Description/MM-Vet pair. No new cohort, benchmark
calibration or paper promotion. Freeze this representation before its GPU
results. The nonuniform COCO study remains a separate, unchanged experiment.

## Label-blind rationale

A processor-only audit (pinned LLaVA-1.5 processor, no model inference) found
that selected text is not necessarily complete in current EviCastKV heads.
This motivates investigating text protection, but **this first diagnostic
does not change selection or reserve new tokens**. First test whether exact
text can be stored more cheaply. No consumed benchmark answer/failure identity
is used to construct or tune this representation.

Capture only each text token's BF16 normalized attention-projection input and
its actual original RoPE phase during the one existing prefill. All heads
share one text input; K/V otherwise duplicate two full-width projections.
Use the audited fused encoded-source projection kernel, directly into the
normal strided per-layer cache lease. Local source-row indices address the
captured original phases; model logical positions are never renumbered.
Store exact scalar residuals for all differences against the existing
decoded cache, and verify three reconstructions bitwise before admission.

Remove only redundant text exact-overlays from the old packet. Visual
protection, low-bit codes, scalar exceptions, mandatory tokens, selected
positions, K/V values and append-only history stay unchanged. Correct LOOK-M
to its ACTUAL merged text K/V, never silently restore unmerged source values.
If source, phases, slot maps, checkpoints and exact residuals are not smaller,
keep the original packet. This common decision rule is offered identically
to all five local methods: EviCastKV, LOOK-M, Prompt Top-K, VL-Cache, PrefixKV.
No method receives an exclusive lossless optimization.

The representation is related to low-rank/source reconstruction (e.g. Palu,
<https://arxiv.org/abs/2407.21118>), but here uses the ORIGINAL normalized
input and exact residual corrections, not a learned/factored approximation
or a Palu reproduction. Cache source history already uses this repository's
audited encoded projection mechanism; this is a text-specific application.
No gradients, weight updates, prototypes or residual repair are enabled.

## Frozen controls and boundaries

Pinned `llava-hf/llava-1.5-7b-hf` at
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, BF16, eager; the same prompts,
media, tokenizer, 0.5 raw source-slot cap, block size one, last-token mandatory
policy, balanced EviCastKV/Top-K and native local allocation otherwise.
Use the existing boundary-eight-bit/middle-three-bit exact-text format.
Every method has anchor and source-storage arms. This is NOT a faithful
native paper reproduction or a shared encoded-byte comparison.

Ordinary FullKV prefill and clone/repack must match the immutable reference.
EviCastKV's anchor must match the previous frozen anchor bitwise. Source
storage must reproduce each method's **complete 128 greedy logit arrays**
and fixed-prefix logits bitwise. Report unchanged 32-token FullKV KL/logit/
token guards honestly, even if an unchanged compressed anchor fails them.
Storage identity cannot turn such a negative quality guard into a pass.

Report actual initial/final/decode-owned peak backing, including source and
all scalar residuals/maps/phases/checkpoints. The fused decode constructor
does not allocate a second raw text K/V buffer. Construction scratch and
inactive FullKV source are excluded by the native diagnostic boundary;
whole-request KV-only peak is NOT certified. Decode timing excludes prefill
and construction. Report conversion and total anchor+conversion construction
time separately; never call their sum a measured full-request speedup.
Inactive comparison formats also remain outside the decode-owned boundary.
These are single-trial, fixed-order diagnostics, not randomized latency
confirmation. Do not use them to pass a no-slower gate.
FullKV-reference ROUGE-L is diagnostic, not independent task-quality evidence.

Complete CPU verification, then two one-A100 workers, 12 CPUs and at most
30 minutes each. Submit after the preceding GPU array finishes, under the
six-A100 aggregate cap. A finite dependent CPU analyzer verifies all twenty
rows and hashes after completion. Preserve failures; no large campaign is
authorized. Only useful storage savings with exact model identity justify
subsequent investigation of complete-text protection on development inputs.
