# Device-record correctness follow-up

This finite diagnostic uses the complete already-consumed sixteen-image component
panel, in its frozen order. Inputs are stripped to IDs, prompts and media. No
reference answers, scores or failure identities enter selection or execution.
It is not a new quality cohort, a benchmark superiority test or an authorization
for a paper campaign. Two one-A100/12-CPU workers, thirty-minute limits, and an
`afterany` CPU reducer preserve failures and planned denominators.

Prerequisite: the two-input device-record benchmark has complete numerical and
storage controls. A strict latency failure remains a failure; it does not prevent
this narrowly scoped correctness audit. Candidate code is byte-identical to
`e3c66f784bef7dfc5f97584e7b847499e24618ac`; only the diagnostic harness is added.

For every input, ordinary common-codec storage-only is compared with synchronous
and device-record guarded hybrids. Require all 128 hybrid logit vectors and all
127 FP64 guard records to match bitwise. Export the synchronous live-parent
counterfactual only in this untimed test and require its logits to match the
ordinary storage parent exactly. Require parent greedy-token agreement and
identical initial/final/decode-owned-peak backing. This does not certify
whole-request KV peak or independent answer quality.

Each loaded model then repeats its first input after seven different requests
(A-B-A isolation). One finite perturbation is injected into the fourth active
final-hidden update in both transports. The unchanged real guard must veto it,
preserve parent tokens and produce identical records and logits. This mechanical
fault test is not a natural repair event or evidence of recovered task quality.
No thresholds, K/V data, source memberships or model weights are changed.

All new output paths are immutable. Existing predictions and paper files remain
untouched. No automatic fresh-cohort or larger-run continuation is permitted.

The first consumed panel attempt exposed a single-head numerical discrepancy,
not a device-record discrepancy. The follow-up saved complete logits on its two
numerical-failure inputs. The full-head counterfactual matched all 127 ordinary
parent vectors exactly, whereas the single-head counterfactual differed at one
prefix each (maximum 0.03125 and 0.125). Neither changed greedy tokens. This is
not grounds to relax bitwise controls. `--full-head-parent` explicitly replaces
the shortcut with the validated full-head counterfactual for both transports,
requiring that bound two-input diagnostic first. Previous single-head timings
must not be silently reused for this corrected configuration. Production code,
thresholds, cache values and source memberships remain unchanged.

The broad full-head retry then passed nine samples but stopped on two
self-contained common-codec packet conversion errors, before those requests'
hybrid execution. The corrected retry binds its new candidate SHA and declares
the standalone-stream repair explicitly. It additionally requires every
common-codec parent logit vector to match the original legacy storage parent;
this representation check is not an equal-byte selection or latency comparison.
Derived failure denominators now include partially completed workers and
explicitly identify unattempted inputs instead of dropping valid partial rows.
