# Complete prototype groups before construction

The standalone-stream repair's second worker completed all eight consumed inputs,
the A-B-A check and an actual finite-perturbation veto. The first worker completed
five inputs and failed during construction of its sixth: the highest-scoring
selected image anchor belonged to a bounded segment with fewer than two free
members. The existing end-of-construction guard correctly refused the short
prototype. No short, padded or mandatory-containing prototype may be accepted.

The correction computes the real descriptor-only assignment plan before tier
construction. If the original best anchor has a complete legal group, retain
that anchor and the same assignment weights exactly. Otherwise discover complete
groups among already selected anchors and choose the highest-scoring feasible
one, retaining deterministic lower-ID ties. Construct exactly one prototype
with the original per-head cap, mandatory policy and byte budget. Reuse the
validated plan rather than recomputing its assignments during construction.
When no complete selected-anchor group exists, fail explicitly; do not alter
exact reservations or silently fall back to storage-only.

This changes only the previously invalid execution path. Unit tests require
every previous valid winning anchor and assignments to remain identical. It
does not establish a new quality improvement. The new immutable correctness
root may reuse completed worker 1 from source `08142be` with its original
manifest and hashes; only failed worker 0 is rerun. The mixed-producer reuse is
explicit and limited to consumed correctness evidence, not a frozen paper
cohort. No references or benchmark scores enter the correction or input choice.

The conditional timing study remains blocked unless all sixteen inputs plus
both workers' veto/isolation controls pass. No latency win is inferred from
successful construction or from the existing CPU-only serialization result.
