# Full-layer restoration before a shared projection-source design

The inverse-phase/capacity study completed all 50 registered records, including
two explicit unsupported PrefixKV phase33 arms. No EviCastKV arm passed both
consumed inputs. Preserve it as negative; no cohort promotion. CPU composition
of its actual logit effects with history repair predicted no passing MM-Vet
combination, so do not run those combinations on GPU.

Before implementing another representation, test whether restoring complete
layers actually fixes the current model-level failure. The frozen diagnostic is:

1. Same two consumed inputs, pinned LLaVA-1.5/BF16/eager, same one prefill and
   unchanged source-only 3-bit EviCastKV anchor. No new independent examples.
2. Replay original FullKV with numerical tolerance <=0.001 and identical greedy
   tokens. Replay the old anchor bitwise. Cache states remain request-private.
3. Restore one ORIGINAL complete layer's unmodified raw BF16 K/V at a time,
   including omitted positions. All 32 KV heads restore together; no per-head
   padding, no phase averaging, no second RoPE. Different layers may have
   different physical lengths, retaining original logical/decode positions.
4. Evaluate every one of the 32 single-layer controls using the unchanged
   32-prefix guard (KL<=0.02, logit difference<=1, token agreement=1). Then test
   exactly the top 2/4/8 layer combinations ranked only by model-logit metrics
   (guard, token mismatch count, max KL, max logit difference, mean KL, layer ID).
   This is a causal diagnostic using saved FullKV prefixes, not an online
   method or evidence from benchmark labels. Keep every result.
5. Independently greedy decode the anchor and the best passing restoration for
   128 predictions, through first EOS for diagnostic FullKV-reference ROUGE-L.
   If none passes, record unchanged-anchor fallback, not a successful change.
6. Charge and report the restored layers' actual initial/final/decode-owned peak
   bytes. These controls generally EXCEED the old cap; they are NOT cap-eligible
   candidates, not direct baseline comparisons, and cannot support superiority.
   No repair, prototypes, quantization change or fabricated low-memory rows.
7. Capture the original full normalized projection inputs to CPU for a bounded
   follow-up cost study. This is explicitly charged diagnostic host storage,
   not a deployed method. Quantizing those inputs at 2/3/4 bits gives measured
   payload lower bounds only: exact overlays, position/phase metadata, execution
   workspaces and construction peaks are not included in those lower bounds.
   Never report that planning bound as implemented cache compression.

If restoring at most eight complete layers cannot improve the actual model
guard on BOTH inputs, do not implement a source codec based only on presumed
benefit. If it does, investigate whether its evidence can be represented inside
the unchanged cap, then verify actual construction and end-to-end model output.

The motivation for a source representation is shared K/V projection information,
related to [Palu](https://arxiv.org/abs/2407.21118) and
[MLA](https://arxiv.org/abs/2405.04434). This diagnostic does not implement either
method, does not train/change model weights, and does not claim joint latent
storage as a new general principle. Any common storage option must also be
offered to applicable baselines before a later controlled comparison.

Two one-A100/12-CPU workers <=30 minutes, complete CPU suite before admission,
finite afterany analysis. Independent quality N=0; fresh and larger campaigns,
paper claims and frozen-candidate promotion remain blocked.
