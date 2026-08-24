# V102 deterministic calibration-ledger review

Date: 2026-08-22

The two V102 calibration ledgers were materialized locally on CPU from clean
commit `b5097f9bfd5770eda29fdfca6c08f7a19c865b6e`. Selection used only pinned
dataset identities, the sealed exclusion manifest, seed `20320821`, and the
preregistered SHA-256 ranking rule. No prompt, answer, reference, score, media,
generated output, or failure identity was inspected.

## Reviewed immutable ledgers

- TextVQA train: 32 unique IDs, 4,854 bytes, mode `0444`, file SHA-256
  `a0667e45729639b1830a308d97d46662f3e6c6c76ddda97749d0663fc329b242`,
  content SHA-256
  `160842ae53f902e08d6e80f17a888ec783a38d4dac8fadbc0e3904471d7edbee`.
- InfoVQA validation: 32 unique IDs, 4,290 bytes, mode `0444`, file SHA-256
  `3308465e540f98dac1003e85a78403968a90ed19bd7d23fe5b2ec6ad8da258c5`,
  content SHA-256
  `6da4d59676cf4def24c474feeef5c3e7fcf01062270f34b68a199a929b42c4fa`.

Each ledger records the exact catalog and exclusion-manifest hashes, candidate
implementation SHA, materialization SHA, protocol SHA, dataset contract,
selection salt/digests, and zero matching exclusions. The TextVQA pool had
34,602 eligible IDs and the InfoVQA pool had 2,801; both selected exactly 32.
The raw ID strings have zero cross-ledger overlap, although scientific
disjointness is defined by the complete configuration-aware identity.

Materialization is consumed and reexecution is disabled. This review
authorizes only an independent configuration-aware disjointness check over the
sealed manifest and these two ledgers. Calibration, GPU or Slurm execution,
candidate freezing, held-out evaluation, and paper claims remain blocked.
