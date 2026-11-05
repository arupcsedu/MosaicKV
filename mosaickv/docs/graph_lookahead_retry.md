# Reservation-boundary retry: not a numerical failure

Both 22498994 workers were killed at 2026-11-05 08:00:00, after 7m51s of their
requested twenty minutes, because the daily bi_fox_dgx reservation ended.
Preserve their incomplete analysis, raw warmups and all successful controls.
Neither worker produced a complete timing screen; no latency claim follows.

All eight audited model controls already passed under adef139: original packet
hash, all 128 own-parent vocabulary vectors bitwise, no fallback, identical
initial/final/decode-owned peak backing, and all 3,906 producer transactions.
This retry changes ONLY controller/provenance/reservation guard/tests. Any
runtime/model/guard change forbids reuse. Bind every measured control's path,
hash and original producer SHA; label it reused, not a new measurement.
Fresh randomized five-warmup/three-repeat full-request timing still executes
every original/optimized reader. Never reuse partial warmup timing as a result.

The finite chain repeats the full CPU suite, runs two one-A100 timing workers,
then terminal CPU analysis. No formal/larger/fresh-quality run is released.
Before submission, defer to daily renewal if twenty minutes plus one-minute
margin cannot fit. Recheck at worker start before heavy inference. Site-local
reservation timestamps explicitly use America/New_York. At most six aggregate
A100s, one GPU/12 CPUs/48 GiB per worker. Numerical and memory gates unchanged.
