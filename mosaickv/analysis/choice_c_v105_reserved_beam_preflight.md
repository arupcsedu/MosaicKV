# V105 reserved label-blind reallocation beam

V104 is sealed negative. TextVQA consumed 23 screening probes and its final
joint probe, leaving no evaluation for the promised post-joint repair. InfoVQA
did retain 11 evaluations, but spent all of them on isolated single-route
swaps. Every InfoVQA swap preserved the greedy token and cleared the KL gate,
yet its maximum logit difference remained between 1.796875 and 1.8125, above
the unchanged 1.0 threshold. It never tested whether two or three individually
helpful swaps would jointly clear that threshold.

V105 changes only the bounded search policy. It reserves 12 of the 24 probes
and two sequential rounds before screening begins. Screening therefore cannot
use more than 11 probes; the next probe is the complete joint cache. If that
joint cache fails, the controller first probes deterministic same-byte
single-route boundary swaps. It ranks those probes using only short-horizon
greedy-token agreement, normalized KL, normalized maximum logit difference,
and deterministic proposal ID. It then constructs all width-two and
width-three combinations inside a three-item beam, subject to the remaining
hard probe ledger.

The beam never reads benchmark labels, answers, references, task scores, or
sample failure identities. It does not change a route's active bytes, the
global active-byte count, mandatory blocks, or atomic layout groups. Every
complete combination uses an isolated cache state and must independently pass
the original two-token, 0.02 KL, 1.0 maximum-logit-difference, and greedy-token
gates. Unprobed combinations cannot be selected, and failed states are
discarded.

V105 requires two new TextVQA-training and two new InfoVQA-validation
identities. The prior V104 exclusion manifest and all four V104 planned IDs
must be excluded before deterministic selection. The reviewed protocol
authorizes only deterministic ID-ledger materialization and two one-A100
preflight workers after this implementation shares one clean immutable commit.
It does not authorize calibration, freezing, or held-out evaluation. Even a
positive four-sample preflight cannot authorize calibration automatically; it
requires a separate review and explicit authorization.
