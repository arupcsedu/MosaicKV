# V9: bounded downstream-sensitivity diagnostic

## Feasibility established before implementation

The V8.1 captures contain six FP16 vectors per layer/prefix: residual input,
normalized attention input, individual attention-head outputs, projected
attention, post-attention residual, and layer output. Both inputs have 32
anchor prefixes and 32 layers. They do **not** contain the anchor's past K/V,
RoPE cos/sin, or effective attention masks. Thus the full downstream Jacobian
cannot be recovered from these captures alone. V8.2 already showed that
treating the remaining decoder as the identity does not give a useful ranker.
Both versions remain negative; their files and consumed identities are retained.

A new diagnostic may capture **only** the same two consumed inputs. At fixed
draft prefixes 1 and 4, capture the anchor's pre-update K/V, actual RoPE
embeddings and masks. These prefixes precede ring overwrite. No new labels,
identities, dataset materialization, calibration, or benchmark campaign are used.

For a single decode prefix, let `f(delta)` replay the frozen decoder from its
captured first-layer input, adding `delta[l]` after layer `l`'s attention and
before its MLP. Past cache tensors are constant; current-token Q/K/V and every
downstream MLP, attention, residual, and normalization are differentiated.
The reference distribution is the original FullKV greedy draft distribution.

The directional KL improvement is

\[
 g_{l,t}^T(-\widehat{\Delta r}_{l,t}),\qquad
 g_{l,t}=\nabla_{\delta_l}
 D_{\rm KL}(p_t^{\rm FullKV}\Vert\operatorname{softmax}(f(\delta)))\big|_0.
\]

One anchor forward plus one reverse vector--Jacobian product (VJP) supplies
all routes' gradients at a prefix, instead of one candidate forward per score.
Cross-check one directional derivative against a genuine forward-mode
Jacobian--vector product (JVP). Normalize its tangent to unit L2 norm so a
near-zero perturbation cannot make the check vacuous; use a fixed first basis
vector if the descriptor's direction is exactly zero. This does not change
proposal scoring. The JVP is a diagnostic cost, not hidden as
free selection. The distinction between a VJP and JVP is explicit.
Use PyTorch's frozen parameters and input derivatives only: no optimizer,
parameter gradients, learned weights, or fitting to benchmark labels.

## Limits of the derivative

The local output perturbation still uses frozen FullKV draft queries/values.
The derivative propagates through the actual downstream decoder, not an
identity tail, but holds earlier-prefix cached states fixed. It is **not** the
Jacobian of the complete 32-token autoregressive trajectory. Finite substitutions
need not follow this first-order approximation. FP16 derivatives are sideband;
the registered FP32 guard and its tolerances remain authoritative.

## Bounded proposer and cost contract

Project each layer gradient through the frozen `o_proj` columns to get head
output sensitivity. Average directional gains at the two registered prefixes.
Generate one-head, one-token, equal-capacity restore/drop pairs with at most
four additions and four removals per route, then take at most 16 positive-gain
route winners. Keep original K/V, mandatory tokens, logical positions and
physical capacity. The ordinary V8 reconstruction pool remains unchanged by
default and is a separately labeled comparator, not silently redefined.

Before probing, require exact anchor-logit replay, finite derivatives,
JVP/VJP consistency and unchanged weights/cache. Bound derivative calculation
to 120 seconds per input; report capture, derivative, JVP and pool times and
private snapshot storage separately. Two captured prompt caches require
additional private storage (roughly two times the 32 MiB prompt allocation,
plus generated-prefix entries); this is **not** a compressed-cache saving.
No measured latency or feasibility success is claimed before execution.

Probe at most 16 proposals through four tokens with immediate rejection at the
first registered violation. Extend only the best two survivors through the
unchanged 32-token guard (`atol=1e-7`, `rtol=1e-6`). Run unchanged clone/repack
controls. Live application KV remains 40 MiB: 32 MiB protected exact prompt
plus 8 MiB generated-history reserve, zero padding. This is not a new claim of
30% smaller storage.

## Stop rule

On each consumed input require at least one accepted, meaningful 32-token
substitution **and** a prospective descriptive screen: correlation >=0.3,
gain over reconstruction >=0.1, positive precision >=0.75 and >=4 predictions.
Screen only identical first-prefix observations; candidate-forward quantities
cannot enter ranking. Both inputs, controls, byte accounting, and provenance
must pass. Otherwise seal V9 negative and stop this substitution strategy;
do not automatically launch another cohort. Even a pass opens only a design
review, not independent benchmark quality claims or fresh evaluation.

Use a finite GPU-array -> `afterany` CPU-analysis chain; no log polling loops.
One A100 and 12 CPUs per worker, two workers, 30-minute limits. Every downstream
experiment permission stays false. Independent benchmark validation would need
a separately registered and authorized protocol.

Primary API references:
[PyTorch JVP](https://docs.pytorch.org/docs/2.8/generated/torch.func.jvp.html),
[PyTorch VJP via autograd.grad](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad.html).
Installed-source contract: Transformers 4.55.2 `LlamaDecoderLayer.forward`,
`LlamaAttention.forward`, `eager_attention_forward`; record its file hash.
