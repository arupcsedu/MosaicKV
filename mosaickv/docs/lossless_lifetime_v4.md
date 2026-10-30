# V4: fair raw comparator and longer exact-storage controls

## Completed predecessor, not selector superiority

V3 array `22181669` completed both workers (7:15 and 7:23); analysis
`22181670` and finite callback `22181957` completed. Every recorded trial
reproduced ordinary exact-cache greedy logits and tokens bitwise. The registered
same-transport overhead gate passed. Its callback correctly did not launch the
conditional profiler. No failed or missing-data GPU retries are needed.

The fused raw control was itself slower than raw separate copies. Report the
encoded method against **both** controls, not only the slower one. Add the
raw/separate comparison retrospectively to V3 using its saved 20 paired
repetitions; preserve V3's original report and registered gate. A new derived
manifest distinguishes the producer and analyzer source SHAs. This additional
analysis cannot become a prospective statistical superiority claim.

## Frozen V4 intervention

Remove repeated PyTorch uint16 view construction from the fused hot path.
Pass BF16 pointers directly and bitcast inside the integer decoder. This does
not change the stored format, values, selection or attention arithmetic.
Strengthen bit controls with signed zeros, infinities and NaN payloads in
history as well as the prompt. Apply the same storage paths to Top-K and
EviCastKV; no selector superiority follows.

Repeat the previous 5-warmup/20-timed-block randomized four-transport study on
the same two consumed inputs and selections. Keep the original gate, and now
prospectively require the encoded/fused upper timing-ratio bound also to be
at most 1.05 against raw/separate. Preserve the same bytes, history policy,
32-token identity gate, and source immutability. Timings still exclude prefill,
selection, debug I/O and reference construction; no request-latency claim.

Add separate **128-token own-greedy** controls for each saved selection:
ordinary exact DynamicCache, raw/separate, and encoded/fused. Require bitwise
logit/token identity, source immutability, all 127 appends, and correct logical
positions. The old FullKV-prefix guard remains separately bound at 32 tokens;
do not extend it using a different trajectory or waive its old failures.

The diagnostic transport's default remains 32 updates. A strict, explicit
integer limit permits longer controls; capacity grows by 16 up to the exact
configured bound. Invalid limits and exhausted limits fail before mutation.
Explicitly drop old history copy-loop variables before leasing a complete
layer. Otherwise Python retains the last old V buffer after the growth
accounting transaction ends. A weak-reference regression verifies its lifetime
at the actual allocation boundary. This matters increasingly at longer history;
the previous raw artifacts remain unchanged and do not certify whole-request
peak. The format and selected/generated values are unaffected.
Record physical initial/final/decode-owned peak bytes; longer raw history may
dilute the prompt compression benefit. Do not assume a minimum saving.

Finally measure bit-exact encoding sizes for the completed history prefix in
those actual caches, retaining a raw mutable tail. This is a **static codec
component measurement**, not an installed history-compression method, a measured
live-cache reduction, or latency improvement. It can inform the next bounded
engineering decision without inspecting human answers.

Whole-request construction/HBM peak, benchmark-quality preservation and
selector superiority remain uncertified. Keep fresh evaluation, calibration,
and paper campaigns disabled. Two one-A100/12-CPU workers, 20-minute limits,
at most six aggregate A100s, scratch-only caches, one immutable source and a
finite CPU `afterany` analysis are sufficient. No continuous polling.
