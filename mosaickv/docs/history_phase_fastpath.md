# Measured native-history host exception cost

The completed whole-request host profile on the two consumed inputs attributed
0.858/0.861 seconds of exclusive instrumented host time to `phase_token`, with
29,632 calls per request. Reading `_version` raises for inference tensors.
The old implementation caught that exception on every call. These instrumented
times are attribution only, not estimates of achievable speedup.

Use `torch.is_inference` to select exactly the previous sentinel without
raising. For ordinary tensors retain the original `_version` counter and
identity/pointer/shape/stride checks. No K/V arithmetic, phase, selection,
mandatory token or storage convention changes. The common history engine may
borrow a contiguous first-128-row view of each existing 192-row prompt buffer;
charge the complete 192-row backing allocation, never just view numel. Shapes
other than the audited 128/192 rows remain unsupported.

Test Source-192 with and without the repaired common native-history reader,
and both optimized PrefixKV readers with and without the same repair. Each
arm must preserve its own parent K/V and all 128 vocabulary vectors bitwise.
Use five excluded warmups and three randomized paired full-request repetitions
only as a point screen, not formal timing acceptance. Actual original-word
history corrections retain the 32-KiB epoch cap and exact storage accounting.
The candidate must still be smaller than every measured PrefixKV control at
initial/final/decode-owned-peak boundaries. Whole-request GPU peak is not
certified. Two consumed inputs imply independent benchmark-quality N=0.

Preserve the earlier native-history latency failure. Promote no fresh cohort
or quality campaign automatically; a screen pass only justifies a separately
registered all-baseline engineering confirmation. If this remains slower,
retain original history and target only a measured remaining dominant phase.
