# Lifetime-history V2: bounded diagnostic successor

EviCastKV and MosaicKV name the same system. V1 executed completely on two
consumed inputs, but quality was mixed and its importance tracking was slower.
This document preregisters V2, not a successful performance result. Original
campaigns and immutable V1 artifacts are unchanged.

## Changes and reasons

1. **Fixed prompt reservation.** V1 reserves only the history positions needed
   by a particular target length. Its short teacher-forced phase can therefore
   retain more prompt evidence than its generation phase. V2 reserves the
   configured history capacity before either phase, irrespective of target
   length. Selected prompt IDs and bytes must match across generation and PPL.
   Only *allocated* history grows on demand. Targets remain inaccessible to the
   selector; complete NLL/token counts are retained.
2. **Prompt-heavy, bounded history.** Reserve 128 rather than 192 generated
   positions, while preserving four generated sinks and a 64-token recent
   window. Test the two frozen hard transaction caps 192/208 MiB. With the
   pinned LLaVA geometry, planning arithmetic gives prompt capacities
   122/137.5 MiB respectively, rather than V1's 90 MiB. These are planning
   quantities, not quality evidence or measured savings. Older important
   generated tokens occupy the remaining history slots; no source values are
   merged, quantized, or offloaded.
3. **Borrowed source snapshot.** An opt-in, read-only typed view replaces the
   redundant full-prefill tensor clone during pure selection/private packing.
   Draft, repair and diagnostic-probe mutation remain forbidden. Source
   references are retired before measured decode. An identical cloned-source
   arm verifies unchanged selected IDs, tokens and NLL; generic isolation
   snapshots retain their historical cloning contract.
4. **Batched importance updates.** On uniform physical layouts, reduce actual
   attention across layers in one batched operation. Refresh a deterministic
   queue of at most 16 eligible victims at once. Sinks/recent positions remain
   protected at every replacement, with original logical positions preserved.
   This is a labeled policy approximation, not the every-token V1 eviction
   rule. Compare against FIFO at identical bytes; it is not presumed better.
   A separate paired ablation protects all mapped question and boundary
   positions before optional visual selection, rather than merely the last
   prompt token. Top-K and EviCastKV receive this mandatory policy identically;
   it cannot be used to claim an unequal-policy win.
5. **Memory scope.** Observe pinned `Cache.update` inputs/owners and
   `torch.cat` outputs inside cache updates, with hooks restored on all exits.
   This supplements registered source/snapshot/packing and growth-copy owners.
   It is not a tensor-shape heuristic, process-memory certificate, or a claim
   about pre-RoPE activations. Also report the separately named whole-request
   PyTorch CUDA-allocator peak (weights/activations included). Do not rename
   post-compression KV peak as whole-request peak.
6. **Observer isolation.** Detailed transaction observation runs in a separate
   private generation replay, which must reproduce every primary token.
   Primary latency excludes that replay. Both observations have immutable
   traces. Timing remains descriptive (one repetition, no warmup), not a
   statistical latency claim.

The sink/recent and heavy-hitter principles are related to
[StreamingLLM](https://arxiv.org/abs/2309.17453) and
[H2O](https://arxiv.org/abs/2306.14048); they are not claimed as novel and this
is not a paper-faithful reimplementation of either.

## Frozen first scope

Use the same first two already-consumed Description/MM-Vet inputs as V1,
without choosing them using outcomes. BF16 pinned LLaVA-1.5, eager HF,
deterministic 1,024-token generation, complete PPL targets, common processors,
prompts/media/seed. Twelve arms per worker:

- FullKV quality reference and native PrefixKV reimplementation control.
- Top-K and EviCastKV FIFO at each 192/208 MiB cap.
- Top-K and EviCastKV batched-importance at 208 MiB.
- EviCastKV append-only at the same 137.5 MiB prompt capacity.
- Cloned-source EviCastKV FIFO at 208 MiB, for borrowing parity.
- Top-K and EviCastKV FIFO at 208 MiB with identical mapped-question protection.

Native PrefixKV controls are different operating points, **not byte-matched
comparisons**. Matched Top-K/EviCastKV pairs require equal measured initial,
final, and growth-inclusive peak KV, not merely equal configured caps.
FullKV is excluded from compressed-method latency comparisons.

Two one-A100/12-CPU workers, at most two hours each, `bi_fox_dgx`, aggregate
maximum six A100s. A finite `afterany` CPU analyzer preserves all failures and
automatically checks completeness, hashes, byte matching, memory replay,
PPL and ROUGE-L separately. Work must run from a clean immutable SHA.

## Recursive-work stopping rules

Do not repeat successful GPU arms, loosen tolerances, inspect benchmark
failure identities for tuning, or launch an unbounded search. No fresh cohort,
powered validation, frozen superiority claim or paper campaign is authorized
by this two-input diagnostic. A candidate is eligible only for a bounded
follow-up on the remaining fourteen already-consumed inputs if it reduces all
three decode storage boundaries and is not worse on either PPL or ROUGE-L
against both native PrefixKV and matched-policy Top-K on both diagnostic tasks.
The CPU callback submits that bounded follow-up automatically; it reuses the
two completed input results and cannot submit a second follow-up. These are
point checks, not statistical evidence.

If no candidate qualifies, preserve the negative result and analyze the
prompt-versus-history and clone-versus-borrow ablations. Another candidate
requires a separately specified change supported by those diagnostics, not
more samples intended to disguise a real quality deficit. A later paper
claim still requires a frozen disjoint powered evaluation and common-budget
curves against the strongest eligible baseline.
