# Actual-query fused reader numerical guard (consumed inputs only)

Both seeded actual-prompt operator checks passed in the preceding immutable
hybrid study. That does not validate real autoregressive queries or history.
Freeze the SAME Triton QK/PV kernels and strict bitwise gate; no rounding fix,
parameter sweep, changed tolerances, new identity, benchmark scoring, or serving
backend promotion is permitted by this finite diagnostic.

Use only the same consumed Description and MM-Vet identities. LLaVA-1.5-7B
revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9, BF16, HF eager, deterministic
128 predictions. The source-precision variant disables all six main mechanisms;
it is not validated full EviCastKV. Baselines are local adaptations, not official
paper-faithful executions. Independent quality N=0.

## Controls, intervention and stopping

1. From a separate immutable worktree, leave the earlier paired benchmark's
   source, jobs and artifacts completely unchanged. Run the complete CPU suite
   with the six recorded submodules and no environment modification.
2. Reverify the previous original/reuse/prefetch/hybrid controls for EviCastKV
   and PrefixKV on each input. Compare logits by matching dtype/shape and UINT8
   words, distinguishing signed zero. Historical torch.equal flags established
   elementwise equality, not that stronger signed-zero property. Keep historical
   artifacts unchanged. Check initial/final bytes against audited companions;
   no new timing or audited-peak claim from these functional rechecks.
3. First run clone/repack controls, then actual fused substitutions at layers
   0/16/31 for ALL first 32 decode prefixes. Capture actual post-RoPE queries and
   K before the parent sequential lease is overwritten with V. The parent eager
   attention runs as an explicit oracle. Clone its V, losslessly encode both
   components, and substitute the candidate attention output/weights into the
   real forward pass. Include actual generated history. Nonzero/padded masks,
   gradients, foreign layouts and unregistered horizons fail explicitly.
4. Require bitwise K/V, weights and attention output at every one of 96 guarded
   operations, and bitwise ALL 128 parent vocabulary outputs at model level.
   Stop on the first mismatch, retain complete scalar traces and first-failure
   tensors, and label numerical rejection, not a successful parent fallback.
   Never fit repairs against the captured queries or weaken the gate.
5. On worker 0 also run A-B-A prefetch requests on ONE loaded model, using both
   consumed identities, fresh private caches, per-input parent word checks,
   correct dataset revisions, and exact activity counters. Compare both A token
   sequences; no cross-request stream or code-scratch reuse may leak.

## Interpretation and follow-up

This guard STILL materializes/clones full K/V and packs them at every guarded
step. These are explicit diagnostic workspaces; a successful result is a
functional substitution, NOT efficient compressed serving or native integration.
It is neither an end-to-end benchmark nor a smaller-cache measurement. Count
clone/packet bytes in traces, retain process allocated/reserved counters, and
do not claim certified whole-request GPU peak. The production reader stays off.
Teacher-forced 593-token PPL is not revalidated by this generation diagnostic.

After the earlier paired timing finishes, run only a CPU retrospective audit
against EVERY tested PrefixKV runtime, not only its reuse arm. Correct for up
to 48 contrasts across both inputs and both own-parent/Prefix comparisons.
Preserve the original registered gate and all failures; stronger retrospective
analysis cannot turn consumed inputs into independent paper evidence.

CPU -> two one-A100/12-CPU/48-GiB, 30-minute workers -> afterany CPU analysis.
At most six aggregate A100s INCLUDING the preceding two-worker benchmark. Common
environment, scratch caches only, clean source and write-once hashed artifacts.
No fresh cohort, calibration, quality sweep, default switch or source edits by
the callback. A model-level success alone requires a separately reviewed
storage-feasible transport before any latency or independent-quality campaign.
