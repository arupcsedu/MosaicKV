# Consumed-input geometry retry and historical archive boundary

The original row-geometry CPU admission failed before submitting GPUs: two
historical tests attempted to read unavailable Git objects at V101/V103 pins.
The remote no longer serves either object. Preserve that failure and all prior
immutable outputs. No production provenance verifier is changed by this repair.

The two tests now explicitly distinguish archive availability. If the exact
blob is present, they retain the original success assertions. If absent, they
assert that the real validator raises a Git error and returns no acceptance or
resolution records. No mock hash, live-file substitution, skip, or permissive
validation is introduced. Passing this rejection branch is not evidence of
historical seal validity. Those historical scans and fresh evaluation remain
blocked. This engineering retry uses only already consumed, independently
hash-bound numerical references, not the unavailable V101/V103 archives.

Run the complete CPU suite from the clean corrective SHA before admitting
the row/head geometry experiment under a new `retry1` root. Keep every source
packet, all-layer reconstructed K/V bits, 128 greedy-token vocabulary logits,
storage bytes, and spill controls unchanged. The follow-up constructor stage
is conditional on safe geometry results and still-negative complete-request
timing. No fresh cohort, quality sweep, benchmark-label optimization, or paper
superiority claim is authorized by this repair.

Use at most two one-A100 workers concurrently for these diagnostics, 12 CPUs
per worker, under the four-A100 aggregate repository limit. A finite dependent
CPU chain analyzes outputs without log polling and stops on unsafe results.
