# Common sequential K/V lease, same-input engineering control

The shared-source variant retains all prompt positions but its larger
reconstruction lease erased persistent savings at the decode-owned peak. Test
one lossless scheduling change, NOT another selector or quality cohort.

Pinned source: Transformers 4.55.2,
`transformers/models/llama/modeling_llama.py:eager_attention_forward`.
For full MHA, `repeat_kv(...,1)` is identity. Native eager attention consumes K
in QK/softmax before consuming V in the weighted-value matmul. Reuse one
private backing buffer: reconstruct K, execute the identical native QK/mask/
FP32-softmax/dropout sequence, then reconstruct V into that buffer and execute
the identical final matmul. Same-stream ordering prevents overwrite before
the last K read. Attention probabilities remain live and are not recomputed.
Do not fuse/reassociate attention operations or change model weights.

Only batch-one single-token eval/inference, full-MHA eager Llama and private
request caches are supported. Reject GQA, other versions/models, gradients,
training, dropout, CUDA graphs, nested contexts and foreign cache tensors.
The temporary operator wrapper restores global HF state on exit; unrelated
model instances continue using the original operator. Existing request-owned
release hooks and failure cleanup remain. Raw generated K/V histories are
separate and append-only, with their full growth allocations charged.

The common component decoder supports raw exact K/V, precision/head-local
packets, bit-corrected selected-text projection, and full-prompt shared-source
packets. Apply this option to applicable baselines too. It does not establish
an EviCastKV algorithm advantage. Quantization/merging/protected-row policies,
selected memberships and logical/decode positions do not change.

First gate: the same two consumed inputs and ONLY the historical source
shadow, plus untouched FullKV retention-one control. For every 33-prefix and
128-greedy decode, run the normal two-buffer transport and this one-buffer
transport. Require BITWISE identical logits, identical initial/final backing
bytes and no larger decode-owned peak. Native FullKV still requires identical
tokens and <=0.001 max-logit deviation. Require A-B-A isolation and immutable
encoded formats. No new source-projection numerical reference is introduced
by this gate; the failed all-source construction remains separately negative.

Full unit/integration suite before two one-A100/12-CPU workers <=30min and
afterany analysis. No fresh cohort, calibration, paper promotion or broader
peak/latency claim. Construction peak remains uncertified, inactive diagnostic
formats are disclosed, zero-warmup one-repetition timing is informational.
