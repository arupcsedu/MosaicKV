# Exact common input RMSNorm replay: bounded execution diagnostic

The predecessor's consumed-input decode host profile attributes 0.50 s
cumulatively to RMSNorm, including 4,032 input normalizations. This is
instrumented attribution, not an expected optimization gain. Capture precisely
the installed Llama RMSNorm operator sequence into private per-layer graphs:
float32 conversion, squared mean, epsilon, rsqrt, multiplication, BF16
conversion, weight multiplication. No fused arithmetic or reassociation.

Use the original module for the first token, then 32 captures and 4,032
replays. Explicit fixed buffers cost 49,156 bytes per request, offered and
charged identically to every method. Reject arbitrary graph allocations,
hooks, training, unsupported models, nested scopes and wrong ordering. Destroy
graphs before retiring their backing owners. Preserve all 128 parent logit
vectors, original cache words, packet hashes, positions and exact storage
enumeration. The common unused-prefill-audit fix also applies to every arm.

Before any model request, a checkpoint-free GPU operator control checks exact
BF16 words and unchanged weights over four fixed input scales. Six arms compare
old/new Source and both old/new Prefix readers, five warmups and three randomized
paired timing trials. The native-history policy is ON for all arms only if the
completed predecessor lowered its own means for all three readers on both
inputs; otherwise it remains OFF for everyone. This rule does not inspect
answers. No broader search is permitted inside the worker.

These are the same two consumed diagnostic inputs, independent quality N=0.
Initial/final encoded bytes must be unchanged by normalization; the entire
49,156-byte allocation must appear in decode-owned peak. Vendor/whole-request
GPU peak remains uncertified. A mean latency screen is not formal acceptance,
and does not authorize fresh cohorts, larger quality or paper claims. The
strict target remains adjusted upper latency ratio <=0.96 against every
original/optimized baseline; <=1.00 is equal-or-faster, <=1.05 trade-off only.
