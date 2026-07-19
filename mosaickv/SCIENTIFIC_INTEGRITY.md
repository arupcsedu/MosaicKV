# Scientific Integrity Policy

## Purpose

MosaicKV is intended to support a research paper and public artifact. Every claim must remain traceable to code, configuration, inputs, and immutable raw evidence. A planned method is not an implemented method; an implemented method is not a validated method; and a validated method has no measured performance until experiments have actually been run.

## Evidence states

Use these terms precisely:

- **Proposed:** described in design or planning documents only.
- **Implemented:** code exists and can be inspected; this alone makes no correctness claim.
- **Validated:** required correctness tests pass at an identified git and config SHA.
- **Measured:** an instrumented run produced raw observations and a complete manifest.
- **Reported:** measured observations passed validation and were transformed into a table or figure by a versioned script.

Do not collapse these states in issues, documentation, commits, or the paper.

## Numerical and tabular integrity

No synthetic number may be placed in a measured-results table. This prohibition includes invented, estimated, projected, interpolated, cosmetically adjusted, manually transcribed without provenance, and placeholder numbers.

- Templates must be labeled **unmeasured template** and use `TBD`, `not_run`, or blank cells rather than numeric placeholders.
- Toy or illustrative values must appear outside measured-result sections and be labeled **illustrative, not measured**.
- Derived statistics are allowed only when computed from linked measured observations by a versioned script. Record the formula and input run IDs.
- Never overwrite raw observations. Corrections create a new derived artifact with an audit trail.
- Do not remove outliers, failed samples, seeds, or benchmarks after observing results unless a preregistered rule applies. Report exclusions and reasons.
- Preserve null and negative results that bear on a claim.

Every table-generation step must reject rows lacking the provenance required by `AGENTS.md` and `REPRODUCIBILITY.md`.

## Fair comparisons

Direct comparisons must use identical prompts, images/videos, tokenization, generation parameters, output lengths, cache budgets, model precision, and backend configuration. The invariant applies to MosaicKV, the full-cache reference, ablations, simple baselines, and published baselines.

Before aggregation, a comparison validator must check stable identifiers or hashes for these fields. A display label or matching filename is not enough. If exact parity is impossible, report the run separately as non-comparable and state what differs; do not use it to support a head-to-head superiority claim.

Budget fairness requires a shared accounting convention. Include all cache payloads and algorithm-required cache metadata. Clearly state whether the budget is expressed in retained token slots, bytes, a ratio to the full cache, or another unit. Do not compare methods using different denominators under one retention-ratio label.

## Full-cache equivalence

Retention ratio `1.0` is a correctness control, not a performance point to optimize away. MosaicKV must reproduce the full-cache reference within the numerical tolerance defined in `REPRODUCIBILITY.md` and registered for the tested precision/backend combination.

The equivalence record must include:

- reference and MosaicKV run IDs;
- compared outputs and tensors;
- absolute and relative tolerances;
- token-level equality status;
- observed maximum and summary errors;
- determinism settings; and
- model, precision, backend, attention implementation, hardware, and software versions.

A failing equivalence check invalidates affected compressed-cache claims until the cause is resolved and all affected results are rerun. Tolerances may not be chosen post hoc to convert a failure into a pass.

## Measurement types

Every experimental row must name its measurement type. Use a small controlled vocabulary and define extensions in the run schema. Recommended values are:

- `reference_measured`: observation from the full-cache reference;
- `method_measured`: observation from MosaicKV;
- `baseline_official_measured`: observation from pinned official baseline code;
- `baseline_reimpl_measured`: observation from a local `*_reimpl` baseline;
- `ablation_measured`: observation from a declared MosaicKV ablation; and
- `derived`: statistic computed solely from linked measured run IDs.

Simulation, projection, and illustrative calculation are not measured types and must not be included in measured-results tables.

## Baseline identity, licensing, and attribution

External baseline source code must be stored under `third_party/` and pinned to a commit SHA. Preserve upstream license and copyright files, include the paper citation and repository URL, and maintain local changes as reviewable patch files or commits. Check redistribution terms before packaging code, models, or data.

Only upstream-authorized code may be called **official**. A local implementation based on a paper must be named `*_reimpl` in code, configuration, manifests, table labels, and prose. Describe it as a paper-faithful reimplementation, never as official code. Record ambiguous paper details, author clarifications if any, deviations, and validation tests.

Do not tune one baseline on the evaluation set more extensively than another without disclosing and controlling the tuning budget. Do not select baseline commits, variants, seeds, or hyperparameters after seeing final benchmark outcomes.

## Authorship and claim review

Before a claim enters the paper or README, a reviewer must be able to trace it through:

1. paper claim or table cell;
2. aggregation artifact and script version;
3. included run IDs;
4. complete run manifests;
5. raw per-example observations; and
6. source, config, model, dataset, and environment revisions.

Material use of external code, datasets, models, ideas, or evaluation services must be cited and comply with their licenses and terms. Automated assistance does not remove author responsibility for inspecting code, verifying evidence, and approving claims.

## Corrections and incident handling

When an error is found, stop propagating the affected result. Record the scope, root cause, affected run IDs and claims, corrective change, and rerun decision. Keep superseded artifacts identifiable rather than silently replacing them. If a released claim changes materially, publish a correction in the same venues as the original artifact where practical.
