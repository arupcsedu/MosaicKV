# Output-preserving draft-gate v1 development seal

This file seals the 16-example COCO-caption development result inspected on
2026-10-15. It is **development-only**, is not held-out evidence, and must not
be used to tune another candidate from its answers, reference scores, or sample
failure identities.

## Immutable evidence

- Run root: `/scratch/djy8hg/runs/mosaickv/output-preserving-draft-edb6edf-20261015-v1`
- Slurm jobs: `21418258_[0-6]` and analysis job `21418259`
- Source base: `edb6edfaf50f4539c9c1969371fefecf0eabccbf`
- Source snapshot: Git-backed development overlay, recorded dirty and content-hashed
- Plan SHA-256: `d97c4b163071e69a40ff566d170d450adc58e0798f30175b66d26b216589337c`
- Input ledger SHA-256: `d389cea234f66b7226f2b1dcc3d380afa1767f3f8bccdd80f446fab11149cb09`
- Analysis report SHA-256: `aab77570386c86366781447335446e0c7d93b9e05db1d1984ced6a29857905ce`
- Aggregates SHA-256: `7d910324d263b4a13297919325dba87f55b050a78feeb0ed15f20a25c6be53b2`
- Cross-curve tests SHA-256: `78a6c9ab2fef43ecbb062a43bc21bb6bf20c7f4db1364fb5df9cf1e4c10ea07f`
- Paired samples SHA-256: `83071d46846d5019b68f269fc893b91322a4c9672db932ee2140ff78683bea22`

All seven GPU arms completed 16/16 unique identities without failed rows. The
retention-one check had token agreement 1.0 and maximum logit difference 0.0.
Every candidate trace recorded isolated four-token draft evidence and exact
initial/final/peak storage accounting.

## Sealed interpretation

At a 25.5 MiB initial cap, the draft-gated candidate and same-budget Prompt
Top-K produced identical answers on all 16 examples. Relative to Prompt Top-K
at 40 MiB, it used 36.25% less initial storage and 30.287% less final/peak
storage. Its CIDEr point difference was +2.285, but the paired 95% confidence
interval was [-2.034, 12.237]. The zero-margin quality gate therefore did not
pass. The result is classified as
`retrospective_development_only_not_paper_evidence` and authorizes no paper
claim or paper-scale campaign.

The draft gate was mechanically active: it rejected hundreds of swaps per
sample and reduced both captured prompt and draft reconstruction error. It did
not change any generated answer relative to the prompt-only two-swap candidate.
No further candidate choice may use these 16 answers or their failure identities.

The separate 128-example protocol in
`configs/experiments/output_preserving_draft_frozen_validation_v1.yaml` freezes
the candidate before execution and excludes this ledger plus the preceding
inspected capacity-sweep ledger.
