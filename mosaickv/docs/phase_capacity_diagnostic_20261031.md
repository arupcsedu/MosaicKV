# Precision-funded prompt capacity (consumed diagnostic only)

Preserve all prior negatives. The online-history repair experiment completed
all 20 rows and passed FullKV numerical controls, but EviCastKV still failed
the MM-Vet token/logit guard. Its extra CPU residual is not free compression.
The saved-logit head+history screen predicted zero passing MM-Vet combinations.
Do not retry those combinations or promote the repaired candidate.

This new, training-free representation/capacity ablation tests a different
mechanism: lower-bit keys in inverse-RoPE coordinates may fund more actual
prompt positions, instead of further optimizing substitutions within an
information-deficient half-prompt cache. Pre-RoPE key quantization is established
in [KVQuant](https://arxiv.org/abs/2401.18079); asymmetric key/channel and
value/token quantization is established in
[KIVI](https://arxiv.org/abs/2402.02750). This is NOT an official implementation
of either paper and does not claim those general principles as novel.

## Fixed experiment

- Same two CONSUMED LLaVA-Description/MM-Vet diagnostic inputs. Independent N=0.
- Same pinned LLaVA-1.5 checkpoint/revision, BF16/eager, prompts, media and seed.
- One prefill only; source cache immutable. No training, label access in
  construction, draft-token emission, prototypes, history repair or offload.
- Offer every representation to EviCastKV, Prompt Top-K and local LOOK-M,
  VL-Cache and PrefixKV reimplementations. Preserve their selected/merged actual
  values before quantization. This is not an official baseline reproduction.
- Five registered arms per method: unchanged post-RoPE 3/3-bit shadow;
  inverse-phase 3/3 at the original 0.5 source ratio; inverse-phase 2/3;
  inverse-phase 2/2; and post-RoPE 2/2 at exactly the last arm's memberships.
- For 2/3 and 2/2 choose the largest feasible ratio from 0.5, 0.6, 0.65, 0.7,
  using actual persistent GPU backing bytes ONLY, before scoring/probing.
  Use the original method's original 3-bit packet cap, unchanged. This is a
  development representation screen with different measured sizes, NOT a
  shared-byte comparison. An infeasible arm is an explicit unsupported row.
- First/last two layers keep their original eight-bit construction. Exact
  text/sink/suffix/scalar overlays remain post-RoPE and are applied AFTER
  phase transport. Existing shared-source text optimization remains unchanged.
- Every positional map and shared full phase tensor is charged. Inverse phase
  is computed from ACTUAL post-RoPE BF16 keys, not falsely described as exact
  native pre-RoPE keys; inverse roundoff is an explicit approximation. LOOK-M's
  already-merged keys use their existing anchor position; no new key merge.
- Reconstruct directly into the charged layer lease, with paired-feature
  in-place rotation. No additional full-key temporary in production decode.
  Original logical/decode positions and raw append-only history stay unchanged.
- Raw retention-1 path bypasses all lossy packet construction. Verify ordinary
  FullKV replay <=0.001 logit difference and greedy identity. Verify GPU rotation
  against the CPU BF16 oracle bitwise, and verify 3-bit shadow trajectories
  against the historical source trajectories bitwise.
- Registered guard remains 32 FullKV-generated conditioning prefixes,
  max FP32 KL<=0.02, max logit difference<=1.0 and token agreement=1.
  Final output is independently greedy decoded for 128 predictions; ROUGE-L is
  FullKV-reference, through first EOS, NOT official MM-Vet judge or PPL.
- Keep negative and unsupported arms. Timings use zero warmups/one repetition
  in fixed order: diagnostic only, never a latency-superiority claim. Initial,
  final and decode-owned peak bytes are reported; whole-request peak remains
  uncertified. More retained tokens can increase the layer lease peak.

Two one-A100/12-CPU workers, at most 30 minutes each, admitted after the complete
CPU suite. Finite afterany CPU analysis. No fresh cohort, calibration, broad
campaign, frozen-candidate promotion or paper claim is authorized by this test.
Advance only if a non-shadow arm passes BOTH inputs' unchanged model guard and
does not worsen either diagnostic ROUGE. Even that would justify only another
separately reviewed validation step, not establish statistical superiority.
