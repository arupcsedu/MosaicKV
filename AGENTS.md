# AGENTS.md

## Scope

These instructions apply to the entire MosaicKV repository. They are mandatory for every coding, evaluation, documentation, and artifact-packaging agent. More specific instructions may add constraints but may not weaken this file.

MosaicKV is a training-free multimodal KV-cache compression system intended for an AAAI 2027 research artifact. Treat scientific correctness, provenance, and reproducibility as release-blocking requirements.

## System contract

MosaicKV consists of all six of the following mechanisms:

1. future-query forecasting;
2. sparse cross-modal evidence graph construction;
3. value-aware block utility;
4. budgeted submodular selection;
5. exact, prototype, and residual cache tiers; and
6. uncertainty-guided residual repair.

Do not silently remove, rename, or substitute one of these mechanisms. If an experiment disables or replaces one, identify it as an ablation or variant in its configuration and result label.

MosaicKV must remain training-free. Do not introduce task-specific gradient updates, learned adapters, calibration training, or benchmark-label fitting under the MosaicKV name. Clearly label any exploratory method that violates this contract as a separate variant.

## Controlled-comparison requirements

Every comparison between MosaicKV, the full-cache reference, an ablation, or a baseline must use identical:

- prompts;
- images and videos, including preprocessing and frame selection;
- tokenization;
- generation parameters;
- output lengths;
- cache budgets;
- model precision; and
- backend configuration.

Pair runs at the example level whenever possible. Cache budgets must use the same accounting boundary and units. Do not compare nominal retention ratios if the methods count different token classes, layers, byte widths, metadata, or tier overheads; normalize them first and document the accounting rule.

If a system cannot satisfy one of these controls, do not put its result in a directly comparable row. Report it separately and disclose the incompatibility.

New PrefixKV, VL-Cache, and LOOK-M unified comparisons must follow
`mosaickv/configs/experiments/shared_byte_comparison_v2.yaml` and
`mosaickv/docs/shared_byte_comparison_v2.md`. Do not mutate historical
nominal-ratio results or rerun that entire campaign. Check the frozen candidate
content and resolved policy fingerprints before preparing a new lane. Bind
actual dataset routes and immutable sample ledgers; unresolved routes block
submission. Require equal measured initial, final, and peak live-cache backing
storage across compressed methods, including padding and preallocation, not
just equal configured caps or tensor-view sizes. FullKV is a quality reference
and is excluded from compressed-method latency comparisons. Preserve failed
rows and retry only missing/failed shards of a compatible immutable plan.

## Full-cache equivalence gate

At retention ratio `1.0`, MosaicKV must reproduce the full-cache reference within an explicitly documented numerical tolerance. The tolerance, compared tensors or outputs, precision, backend, attention implementation, determinism settings, and test procedure must be recorded before accepting results.

The `1.0` path must not perform lossy prototype conversion, eviction, or residual approximation. A failed equivalence check blocks quality and systems claims for the affected configuration. Never loosen a tolerance after seeing a failure without documenting the reason and rerunning both reference and candidate results.

See `mosaickv/REPRODUCIBILITY.md` for the repository's default equivalence contract.

## Results and scientific integrity

No synthetic, invented, estimated, placeholder, or interpolated number may be placed in a measured-results table. Empty planned tables must use nonnumeric markers such as `TBD` and must be labeled as templates, not results. Illustrative numbers belong only in clearly labeled examples that cannot be mistaken for empirical findings.

Do not claim that a milestone, baseline, integration, dataset run, or result is complete without inspectable evidence. Preserve failures, exclusions, and negative findings. Do not selectively omit seeds or examples after inspecting their outcomes.

Every experimental row must record, directly or through an immutable run manifest:

- git SHA;
- config SHA;
- model ID and revision;
- dataset and revision;
- CUDA version;
- NVIDIA driver version;
- PyTorch version;
- Transformers version;
- vLLM version;
- SGLang version;
- GPU type and count;
- backend and attention implementation;
- seed; and
- measurement type.

Use an explicit value such as `not_used` when a framework is not involved; never leave required provenance ambiguous. Record whether the worktree was dirty, and preserve the patch or refuse to publish the run as canonical.

Follow `mosaickv/SCIENTIFIC_INTEGRITY.md` and `mosaickv/REPRODUCIBILITY.md` for result classification, manifests, validation, and reporting.

## Baseline provenance

External baseline code must live under `third_party/` and be pinned to an immutable commit SHA. Preserve its license, copyright notices, citation information, and attribution. Record local patches separately; do not overwrite upstream history or imply that patched behavior is upstream behavior.

Paper-faithful reimplementations must have names ending in `*_reimpl`. They must never be described as official code. Their documentation and result labels must identify the paper implemented, deviations or ambiguities, validation evidence, and the fact that the implementation is local.

Do not vendor model weights, datasets, or code whose license does not permit redistribution.

## Engineering workflow

- For the user's October 26 event-driven workflow, chain finite Slurm jobs
  with `afterany` so failed producers also receive completeness/error analysis.
  Use write-once decisions and explicitly bounded follow-up scope; do not
  continuously poll queue state or tail logs. The readout-audit completion
  example runs one constrained, ephemeral Codex CLI review, not an automatic
  wake-up of the interactive chat or authority to widen experiment permissions.

- The user's October 26, 2026 instruction authorizes scoped local commits
  without repeated consent. Consolidate related changes and avoid commits less
  than two hours apart unless the user explicitly requests an immediate commit.
  This does not authorize pushes, fresh cohorts, or larger experiment campaigns.

- The user's October 23, 2026 validation request permits at most six A100s,
  in aggregate across MosaicKV jobs (not per array). This supersedes the old
  four-GPU limit for this validation. Preserve running jobs; if occupancy is above
  six, admit no new workers until it falls below six. Register future arrays
  with the tested schedule controller;
  use `maximum_a100s: 6` and `six_a100s_explicitly_authorized: true` in its policy.
  The controller defaults to four for historical policies. Do not change other users' jobs.

- Inspect relevant code, configuration, tests, and repository status before editing.
- Use only the common environment at `/scratch/djy8hg/env/mosaickv` for HF,
  vLLM, SGLang, evaluation, and development runs.
- Source `mosaickv/scripts/cache_env.sh` before any setup, test, model access,
  server, container, or evaluation command. Every cache and temporary path
  must be rooted outside home, normally under
  `/scratch/djy8hg/cache/mosaickv`. Never persist `HF_TOKEN`.
- Canonical validation, benchmark, parity, and paper-result runs require a
  clean tracked and untracked worktree. Run
  `mosaickv/scripts/assert_clean_worktree.sh` first. A dirty-tree run is
  exploratory even when its patch is preserved.
- Do not import AAFLOW through a sibling checkout, `PYTHONPATH`, or dynamic
  path mutation. MosaicKV currently has no AAFLOW runtime dependency. If code
  is intentionally reused later, isolate it under a pinned and licensed
  `third_party/` checkout or a clearly attributed MosaicKV compatibility
  namespace, and import it through MosaicKV's package boundary.
- Keep algorithm logic, backend adapters, evaluation code, and result presentation separable.
- Put all behavior-affecting choices in versioned configuration; do not rely on hidden notebook state or unrecorded environment variables.
- Derive the config SHA from the canonical resolved configuration, not merely its filename.
- Add correctness tests before treating optimizations or backend integrations as validated.
- Use fixed seeds where supported and record nondeterministic operations where they are not.
- Never edit raw run artifacts by hand. Derive aggregate tables from immutable per-run records with scripted transformations.
- Do not commit secrets, access tokens, private dataset contents, generated model weights, or large untracked artifacts.
- Preserve unrelated user changes and keep changes within the requested scope.

The repository contains governance, research infrastructure, and partial MosaicKV mechanism implementations. Verify implementation and correctness evidence in source and tests; do not infer completion or measured performance from plans, namespace placeholders, configuration fields, or design descriptions.
