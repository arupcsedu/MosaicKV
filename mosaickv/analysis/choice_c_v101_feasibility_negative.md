# Choice C V101 exclusion feasibility review

Status: **sealed feasibility-negative; no calibration samples were materialized**

## Decision

V101 cannot proceed under its preregistered DocVQA-validation and
TextVQA-validation contracts. The immutable exclusion snapshot covers every
example in both pinned validation splits, leaving zero eligible IDs. The
materialization, disjointness-check, calibration, freeze, and held-out stages
remain disabled.

This is a protocol-feasibility result, not an algorithm-quality result. It says
nothing about MosaicKV accuracy, memory reduction, or latency.

## Sealed exclusion evidence

- Manifest: `/scratch/djy8hg/runs/mosaickv/choice-c-v101-v100-fresh-disjoint-calibration-v1/inputs/exclusion_manifest.json`
- File SHA-256: `39a068c8863ea4afd936ce698d933fff5abed31e6f43d1874f778039ef139ca2`
- Internal content SHA-256: `ea217365356ea199de1080245618c12502857be4071f001e0fbbd3ce0a06a91b`
- Scan source Git SHA: `36659a51354e48c3ad8bdaa983ef530ba15fe842`
- Source worktree dirty: `false`
- File mode: `0444`
- Bound source records: 2,410
- Unique excluded identities: 10,349
- Every bound source hash was revalidated after the scan.

## Split coverage

| Task | Pinned split | Split size | Excluded IDs | Eligible IDs | Coverage |
|---|---|---:|---:|---:|---:|
| DocVQA | validation | 5,349 | 5,349 | 0 | 100% |
| TextVQA | validation | 5,000 | 5,000 | 0 | 100% |

The split sizes were checked against cached `dataset_info.json` metadata at the
exact pinned dataset revisions. The review did not load media, prompts,
answers, references, task scores, generated tokens, or failure identities.

## Why materialization is forbidden

V101 requires exclusion of every previously observed sample and deterministic
selection from the remaining IDs. Because the eligible set is empty for both
tasks, requesting 32 samples per task must fail. Reusing validation examples,
silently weakening the exclusion boundary, or switching splits after seeing
the result would violate the preregistration.

## Follow-on protocol

V102 is separately preregistered around two different candidate pools:

1. TextVQA `train` at the same pinned repository revision.
2. InfoVQA `validation` from the `InfographicVQA` configuration of the pinned
   `lmms-lab-encoder/DocVQA` repository.

The V102 pools remain unmaterialized. A new configuration-aware identity
schema and a formal exclusion scan must be reviewed before any materialization
can be authorized.
