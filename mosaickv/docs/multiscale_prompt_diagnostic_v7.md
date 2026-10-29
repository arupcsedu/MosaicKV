# V7: multiscale prompt evidence, consumed-input diagnostic

## Causal motivation

V5 held the exact selected prompt fixed while increasing history capacity from
128 to 512 positions. Both tasks' generated ROUGE-L remained unchanged. MM-Vet
PPL improved but still missed native PrefixKV. Therefore more history alone
cannot fix the observed generated-answer deficit; stop increasing history caps.

Native PrefixKV uses head-averaged full-prompt query attention, whereas the
frozen exact candidate uses a recent prompt window. Full-prompt query evidence
may retain visual/global context missing from that window. This is a hypothesis,
not a guarantee or a novel claim about using full-prompt attention.

## Evidence-only mixture

For each layer and query head, let `A[q,i]` be original eager prompt attention.
For the last `w` query positions, capture the synthetic selection distribution

`E[q,i] = (1-alpha) * A[q,i] + alpha * mean_{all prompt q'} A[q',i]`.

Do not pool query heads before constructing this mixture. At alpha below one,
individual window-query differences remain with factor `1-alpha`. Alpha one is
an explicitly registered full-prompt-only scoring ablation. This is selection
evidence, **not** a claim that the model actually attended according to E.

The forward hook reads the original eager probabilities, computes the full-query
mean on device and stores only a private bounded-window mixture. It returns no
replacement attention tensor to the model. Original prefill logits, K/V and model
weights must remain exactly unchanged. The private capture does not retain extra
full Q-by-K maps across layers. This does not claim that the underlying HF
`output_attentions=True` result collector avoids its existing raw-map retention;
whole-process allocation must still be measured. Prompt-derived evidence only:
no answers, references, target lengths,
model probes or training. Existing graph, utility and exact selection consume
the declared mixed evidence; raw query projections are unchanged.

The new setting is explicit, validated, included in config hashes only when
nonzero and in effective variant labels. Zero remains exactly the old capture
path and old config identities are preserved. Retention one bypasses lossy
selection and this evidence modification.

## Frozen diagnostic scope

Run only the same two consumed Description/MM-Vet inputs. Keep the V5 profile,
mandatory policy, history FIFO behavior, BF16 LLaVA checkpoint, eager backend,
forced 1,024-token output and compatible frozen FullKV/native references.
Test EviCastKV alpha 0.25/0.5/1.0 at both V5 endpoint capacities (128 and 512
history positions, corresponding to 223/421 MiB hard transaction caps).

Reuse the four frozen **window-only Prompt Top-K** controls from V5, with identical
actual initial/final/working-peak bytes. Separately run full-prompt-only Top-K at
both endpoints as a scoring ablation; never silently relabel it ordinary window
Top-K. Sixteen new one-arm shards, one A100 and 12 CPUs each, maximum six
concurrent, 30 minutes and 16 GiB host RAM each. Start only after V6's zero-clone
equivalence and ownership verification passes.

Keep the point gates unchanged: exact matched realized bytes; smaller initial,
final and active decode-transaction storage than native's distinct operating
point; non-worse PPL and ROUGE-L separately versus native and frozen window-only
Top-K. N=1 per task cannot establish statistical preservation. Report every
configuration including negative results, and keep timing descriptive/unpaired.
The full-prompt-only Top-K rows diagnose whether gains come from evidence rather
than the graph; they cannot be omitted from the result table.

Global registered KV ownership, working decode transactions and CUDA-allocator
peak remain separately labeled. No generic whole-request KV-peak certificate,
fresh cohort, calibration or paper-scale promotion is authorized by this study.
