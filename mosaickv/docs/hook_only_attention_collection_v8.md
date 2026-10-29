# V8: avoid duplicate model-output attention collection

Engineering optimization only; not a new evidence-selection policy. The audited
installed Transformers 4.55.2 implementation is:

- `transformers/models/llama/modeling_llama.py:LlamaAttention.forward`, which
  always returns eager attention weights to module hooks;
- `LlamaDecoderLayer.forward`, which consumes the attention output, independent
  of whether a model-level collector returns those weights to the caller.

MosaicKV already installs its own capture hooks and owns the required private
window or full map. Asking HF to additionally return `output.attentions` retains
the original full Q-by-K maps until the complete prefill output is released.
For exactly the pinned 4.55.2 `LlamaAttention` classes in eager mode, suppress
that redundant model-output collector. Do not change the attention computation,
private capture, masks, logits, K/V or model weights. Other architectures,
custom subclasses and other library versions retain their existing behavior.
This applies to every method using the audited adapter, not just EviCastKV.

CPU verification compares the original forced public collection path with the
private-hook path and requires exact logits, raw captured probabilities and K/V.
Trace both the private capture and model-collector policies explicitly.

GPU verification uses exactly the four consumed V6 controls, with identical
configs and byte caps. Require exact answers, all generation/teacher-forcing
tokens, NLL/counts, PPL, ROUGE-L, selected blocks and live-KV storage. Registered
KV owner peak must remain unchanged; whole-request PyTorch CUDA-allocator peak
must decrease. This is an activation/output-retention improvement, **not** a new
active-KV compression gain or generic KV transient certificate. Single-trial
latency remains descriptive, with negative results retained.

Four one-A100/12-CPU workers, maximum six aggregate, 30-minute limits. Submit
only after the other diagnostic GPU array completes, then use finite afterany
analysis. No calibration, fresh identities or larger experiment is authorized.
