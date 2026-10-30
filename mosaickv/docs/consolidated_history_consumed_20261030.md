# Consolidated exact source history — consumed-input engineering variant

Source-history decode reconstructs up to seven separate16-token archives per
layer at output128. This variant joins their encoded normalized inputs into
one contiguous source packet. Seven low-occupancy projection launches become
one larger grid, reusing the identical fixed16-row projection arithmetic.
No K/V value, original RoPE phase, selected position or history token changes.
All original phases and logical positions are carried in chronological order.

Code/bases/sign-mantissas/literal streams concatenate at existing1024-value
boundaries; literal offsets are translated, never approximated. Existing
int32 correction locations are remapped head-locally. Three independent
reconstructions compare every original K/V bit before a layer publishes the
replacement. Later failure leaves earlier exact replacements and untouched
later layers semantically valid; the failed request is not promoted.

Build one layer at a time. Audit original/new packets, reference K/V and
constructor overlap; do not keep32 old/new packets simultaneously. Rebuild
the audit exclusion set when old allocations are freed to prevent recycled
addresses hiding new temporaries. Original phase owners are freed only when
no archive needs them and a canonical concatenated phase registry owns them.

Keep original HF eager attention and contiguous attention leases. No CUDA
graphs or alternate attention arithmetic in this variant. Retention-one and
selected128-token logits must remain bitwise equal; same two consumed inputs,
all six paths, five warmups and twenty randomized technical paired repeats.
Initial bytes must match. Final storage <=1.02x and observed decode-owned peak
<=1.05x reference are preregistered engineering bounds, not an assertion that
larger bytes are a compression win. Report exact bytes and latency, including
all construction/coalescing checks in decode. Require real measured memory
and quality superiority independently before a paper claim.

Fresh cohorts, calibration and paper campaigns remain disabled. CPU tests
cover every BF16 pattern, mixed raw/encoded streams, unique head-local patch
translation, unchanged history/positions, owner accounting and partial-failure
validity. GPU admission follows the full CPU suite and the earlier GPU array
finishing, keeping aggregate use <=6 A100s. Canary success may release only
the finite12-worker method/input comparison, never a new evaluation campaign.
