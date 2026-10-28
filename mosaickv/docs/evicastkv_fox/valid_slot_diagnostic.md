# Targeted, evaluation-only valid-slot execution

The consumed native MileBench captures locate the first difference at valid
attention probabilities (layer 0) or the attention context (layer 5), before
downstream residual amplification. This justifies one bounded execution
experiment, not another broad precision sweep or a quality-cohort revision.

For batch-one, one-query eager attention with the same binary validity pattern
for all heads, gather valid K/V slots in their original order, call the installed
eager implementation without changing its precision, and scatter returned
probabilities back onto the original physical axis. Stored K/V tensors, logical
positions and physical-cache accounting are unchanged. The temporary attention
operands are explicitly not new compressed-cache storage. Head-dependent masks
and finite attention biases remain unsupported for this narrow diagnosis.

The all-valid path calls the installed function with the original object
identities; it does not gather or change its operands. The wrapper is scoped to
one evaluation adapter and removed on exit. No installed package or weights are
edited. The production runtime never enables it by default.

Test only the same two consumed native-failure inputs. Keep sixteen forced
decode steps, the original eight-FP16-epsilon normwise bound and 1e-4
retention-one/generate parity threshold. Preserve native captures as negative
evidence. A pass qualifies only this **new transport variant**; it does not
reclassify the old native padding control, resolve the separate COCO A/B
projection-geometry failure, certify fresh media, or authorize confirmation.
Before adoption, every compared method would require a common-policy regate;
timings must include any gather/probability-scatter overhead. No compression,
quality or latency benefit is established by numerical identity alone.
