# Async same-value cached-text/native construction

The inline verified handoff passed words but was slower than same-buffer
construction. The common ownership-catalog correction passed words/storage,
but the all-input 4% point gate remained negative (one input passed, one did
not). Retain both findings and all common baseline optimizations.

The existing native fitter already reconstructs each original parent K and V
component. Reuse that exact component for compact cached-text rows, rather
than reconstructing full cached-text K/V separately on the foreground thread.
No source precision, selected position, protected value, model weight, RoPE
phase, generation parameter or numerical tolerance changes. This is default
off as `verified_constructor=async_handoff`, on the two consumed full-MHA
600/604-token inputs only. All six primary mechanisms remain disabled in this
precision/storage diagnostic; do not call it complete-algorithm validation.

One private worker/stream/native handle and at most two pending layers remain.
Source ready events precede reads. Compact text owns copied rows; no full
reference survives a component. Original finite/close/protected checks,
direct native word validation and original stride audits remain. Audit the
complete outer cache at 0/1/127 extra positions too. Keep 32-KiB/layer and
one-MiB/request numerical correction caps. These are execution corrections,
not residual repair.

Only the main thread publishes ready cached-text packets into the same cache.
It waits on explicit completion events, records cross-stream ownership and
checks old packet identity/order. Publish progressively, so completed outputs
do not retain all original-plus-text packets until prefill ends. Reconcile
the common layer/extra ownership catalog independently before retiring prefill.
Consume each future once; do not recompute on failure or fall back silently.
Retire the worker and handle before decode; no state crosses requests.

Before native substitution, require the exact frozen preconversion packet
digest. Require all 128 parent vocabulary-logit vectors bitwise, no FullKV
fallback and explicit queue/callback/publication/retirement counts. Original
control storage must be identical; candidate initial/final/decode-owned peak
storage may change only by counted nonincreasing native corrections. Input
text that was protected before stripping must not acquire spurious repairs.

Record main-thread complete-request timing, never add overlapping worker
time. Temporary auditing describes the worker's observed tensors and retained
repairs, not other pending text outputs or whole-request GPU peak. The latter
remains uncertified. No new workspace may be hidden from persistent/decode
backing counts. No independent quality claim follows parent-bit parity.

CPU isolation/order/owner/packet tests precede two one-A100 workers. Each worker
audits all arms BEFORE five warmups and three paired timed repetitions. Use
12 CPUs and a twenty-minute limit, then afterany terminal CPU analysis. Keep
the all-input <=0.96 point screen and the subsequent adjusted twenty-repeat
gate distinct. No fresh cohort, quality sweep or paper campaign is released.
