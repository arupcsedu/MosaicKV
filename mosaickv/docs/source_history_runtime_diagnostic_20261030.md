# Source-sealed history runtime diagnostic

This separately versioned opt-in follows the positive, consumed-input source
packet feasibility. It is an exact storage variant, not a selector revision or
paper campaign. Existing selected prompt K/V and their logical positions are
unchanged. The normal CLI and all existing `run` callers remain off by default.

Enable only for a fixed 128-token generation diagnostic. Thirty-two-token
feasibility was negative because the extra normalized tail outweighed saving.

Each layer retains its existing raw sixteen-position K/V tail and an additional
sixteen-position normalized-input tail. A request-local hook captures the actual
input before the unchanged projections. Original rotary tensors and logical
positions are shared only after verifying that every layer receives the same
original tensor objects. All additional tail storage is charged.

At a full tail, prepare exact source packets for every layer, retaining the
original RoPE phases and exact sparse bit corrections. Check reconstructed K/V
three times. Commit the entire seal only after every layer passes; errors leave
the existing tails untouched. Packets never alias the mutable tails. Layer
leases reconstruct sealed history into the original chronological positions.
The cache's scalar accounting must match independent tensor-backing enumeration.
The peak audit includes observed tensor results during source packet construction.
Weights, attention activations, internal workspaces and request prefill remain
outside this declared decode-owned boundary; whole-request peak is uncertified.

Finite execution: first two consumed EviCastKV canaries, then at most twelve
one-A100 workers (six paths by two consumed inputs), at most six GPUs in aggregate,
twelve CPUs per worker and thirty-minute limits. Five warmups and twenty
randomized paired repetitions measure both ordinary and source-sealed history
on the same GPU. FullKV is excluded from latency comparison; its storage row is
an all-position lossless control, not ordinary uncompressed FullKV.

Advance from the canary only when all original 128-token logits and their
32-token prefix are bitwise unchanged, initial bytes match and measured final
and observed decode-owned peak bytes both decrease. The two-stage chain uses
afterany analyses. Failures stop advancement without weakening thresholds.
Latency is reported even when worse. No fresh inputs or baseline-quality claims
are authorized. PrefixKV/VL-Cache rows remain local append-only adaptations;
LOOK-M preserves its actual merged prompt K/V and receives identical lossless
generated-history treatment.
