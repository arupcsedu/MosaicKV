# Common, explicitly rounded BF16 RoPE

The saved joint-KV cProfile attributes approximately 0.58 seconds to 4,032
steady-state RoPE calls (not pure removable GPU time). Coalesce the original
query/key elementwise operations into one launch. Explicitly round each product
to BF16 before adding; disable FMA. Negate rotated BF16 words with a sign-bit
operation, preserving signed zero. Do not fuse the two products into a different
rounding expression. The reference is installed Transformers 4.55.2
`Llama.apply_rotary_pos_emb` and `rotate_half` (Apache-2.0).

Default-off private one-token BF16 full-MHA Llama geometry only. The same
implementation applies to every compared baseline. Outputs retain the native
Q/K geometry and lifetimes; there is no persistent extra GPU tensor. No weights,
positions, selected values or approximate representations change.

Before model loading, compare full-width GPU reference words at zero, tiny,
ordinary and large magnitudes, including negative zero. Reject spills, scratch,
unsupported geometry or a single word difference. Then require all 128 parent
vocabulary vectors bitwise, original packet digests, exact own-control initial,
final and decode-owned peak bytes, 4,064 calls and restored module methods.

After the preceding replay and common epilogue diagnostics finish, freeze whether
to retain the common epilogue: retain it for all methods only if all three paired
own-control means improve on both consumed inputs and its accounting/numerical
checks pass. Otherwise keep it off for every method. This engineering choice
uses consumed timing, never benchmark answers. Compare six arms (each source
and two PrefixKV readers, with/without coalesced RoPE), five warmups and three
randomized timed requests. Two one-A100 workers, 12 CPUs and 20-minute limits,
full CPU suite before GPU release and automatic afterany analysis.

Three trials do not establish formal 4% acceptance. Independent benchmark N=0,
whole-request/vendor peak uncertified, no fresh/larger quality campaign, and the
six-mechanism-disabled storage variant remains distinct from full EviCastKV.
