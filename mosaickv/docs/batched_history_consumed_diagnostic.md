# Bounded batched history sealing

This is an exact storage/latency experiment on the same two consumed inputs.
There is no selection change, quantization, eviction, label access, fresh
cohort, or automatic calibration/paper promotion.

The existing implementation seals each K and V tail independently and needs
one host synchronization for each variable-length literal allocation. The new
coordinator batches at most four layers (eight K/V tensors), reducing 64
allocation-size synchronizations to eight per seal on the 32-layer model.
It prepares all immutable bodies before resetting any layer's tail counter.
Codec-block-aligned views share a global literal pool. Both CPU and GPU
decoders retain every BF16 bit, including all NaN payloads. Actual backing is
charged once through the cache's shared-owner list, independently checked
against all live layer buffers; view sizes never substitute for backing sizes.

Four-layer staging bounds the temporary raw batch at two MiB for the registered
checkpoint. Scratch observation includes pending encoded groups and their
raw staging buffers, not merely the completed history. Original K/V, logical
positions, mandatory tokens and append-only history remain unchanged.

All six primary methods receive controls at 32 and 128 tokens. PrefixKV and
VL-Cache remain explicitly append-only local adaptations; LOOK-M's merged
values are preserved. The common prompt format chooses minimum actual bytes
among validated dual projection, direct fused compact replay and encoded K/V, not the slower original
full-shape replay. This can cost slightly more metadata than the historical
minimum-storage format; actual bytes are reported. EviCastKV retains its
registered 62 MiB initial cap and 96 MiB observed decode-owned 32-token peak
cap. No 96 MiB cap is asserted for 128-token append-only growth.

Five warmups and twenty randomized paired timing blocks include both history
policies for all five compressed methods on one GPU per input. FullKV is a
quality reference only. Timing excludes prefill, selection, representation
construction and trace I/O. It is not an end-to-end latency claim. Whole-request
HBM peak remains uncertified. A peak regression is explicitly reported, not
concealed by a faster median.
