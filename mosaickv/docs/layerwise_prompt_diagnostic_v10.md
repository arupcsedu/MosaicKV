# V10: bounded layer-wise shared-prior ablation

EviCastKV and MosaicKV denote the same system. Preserve V2–V9 as negative
development evidence. No fresh samples, calibration, paper campaign or automatic
follow-up are authorized by this protocol.

## Rationale and fixed schedules

V9's mixed prior preserved generated-answer ROUGE better, whereas its full
shared prior reduced long teacher-forced PPL. These are complementary signals
on consumed inputs, not proof that combining them will succeed. Test layer
heterogeneity once, without fitting layer policies to answers or references.

For attention A, each layer uses the already implemented evidence formula
E[h,q,i]=(1-alpha_l)A[h,q,i]+alpha_l mean_q'(mean_h A[h,q',i]).
The prior uses pinned PrefixKV's dtype/reduction order; its general scoring
principle is not a new contribution. Only private selection evidence changes;
model forward attentions/logits, original K/V and logical positions are unchanged.

Four schedules are fixed before GPU execution. Assign alpha=1 to (a) the early
half of layers; (b) the late half; (c) the half with highest retention weights
1-forget in the pinned profile; (d) the complementary profile half. Other layers
use alpha=0.5. Resolve ties by layer index. No benchmark labels enter schedules.

Use precisely the existing one Description and one MM-Vet consumed identity.
At history capacity 256 and the existing 289 MiB complete decode-transaction
cap, run four schedules plus three contemporary controls per input: ordinary
window Top-K, uniform shared-prior EviCastKV 0.5, and uniform shared-prior 1.
This is fourteen one-arm workers, one A100/12 CPUs, 16 GiB RAM, 30-minute limit,
maximum six A100s aggregate in the isolated source's authorized scheduler policy.
Use reservation bi_fox_dgx. Analysis uses finite afterany dependency. No log
polling loop or self-expanding GPU experiment chain is installed.

Require controls to reproduce frozen configs, tokens, NLL/counts, PPL and ROUGE
exactly. Require measured initial/final/decode-transaction storage to match
contemporary Top-K. Require both quality metrics to be non-worse than native
PrefixKV and matched Top-K on each input. Native PrefixKV has a different
operating point; it is not a byte-matched row. Record all negative schedules.
N=1 per task cannot establish statistical quality or latency superiority.

## Saved-selection and memory audit

V9 saved whole-selection hashes, per-layer capacities and prompt-mass summaries,
but deliberately omitted per-head position catalogs. Those hashes establish
that selections differ; they cannot identify which layers differ. Do not
invent that attribution. The private memory replay in V10 adds streaming
per-layer/head position fingerprints (no raw K/V). Compare all layers against
both contemporary uniform policies. Trace work is excluded from primary latency.

Report initial/final physical backing storage; active decode-transaction peak
including replacement buffers; registered request-owner peak; and the process
CUDA allocator peak separately. The owner ledger deduplicates aliases and uses
weak references, but is not an exhaustive prefill allocator audit. Do not sum
role totals that share backing storage. Native control retains inactive source
snapshots whereas bounded variants retire them: that comparison cannot establish
an algorithmic whole-request peak win. CUDA allocator memory includes weights
and activations. Exact whole-request KV peak remains explicitly uncertified;
the CPU audit documents coverage and gaps rather than issuing a false certificate.

Only a meaningful point-level pass can justify proposing a frozen disjoint
validation protocol. It cannot automatically freeze a statistically validated
candidate or authorize the paper campaign.
