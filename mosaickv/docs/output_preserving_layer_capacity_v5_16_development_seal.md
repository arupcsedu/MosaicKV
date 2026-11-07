# V5 layer-capacity canary seal

V5 is sealed as an incomplete negative, development-only result. All 16
planned identities are consumed and must not appear in calibration, powered
validation, or paper evidence. No sample-level answer, reference, score, or
failure identity from this cohort may be used to tune a successor.

The FullKV and two Prompt Top-K workers completed all 16 rows. The V5 worker
reached only 4/16 rows before its two-hour limit, so the dependent analysis
correctly failed closed and withheld aggregate quality metrics. The completed
rows established correct physical accounting but exposed two independent
blocking failures:

- initial cache: `25.5 MiB` versus the comparator's `40 MiB`;
- final and peak cache: `33.375 MiB` versus `47.875 MiB`;
- minimum observed boundary reduction: `30.287206266318534%`;
- physical padding: zero;
- logit-stability overhead: `1271.2865--1381.557625` seconds per completed
  measured sample;
- all four completed requests selected the Prompt Top-K anchor because none
  of the eight layer-transfer candidates passed the fixed KL/logit thresholds.

Increasing the time limit would therefore finish an ineffective candidate; it
would not validate V5. A successor must use new identities, preserve the fixed
thresholds and byte cap, and eliminate the preliminary model-probed
route-impact rounds. The V5 outputs remain immutable and are not paper
evidence.

## Immutable artifact bindings

- plan: `cb651bc91a23acf19d14c47c707b382bd0077d5eee73f7ef140fde4fc3448015`
- ledger: `938563b5eb5de742b30d8171d333cd75298fbda6acfad60418e2e61cd123e2b8`
- fail-closed report: `16d0904f0ab35ee548aadf2cf0fca52923ce87b75e7a1a34bd3bb2f43b662c0d`
- partial candidate rows: `5cc39fff64045d4ff43f6e6af5441def231f6adfd7c69e190d2aa28cfe9c3a7d`
- Slurm array: `21475539` (`_3` timed out)
- controller: `21475540`
- analysis: `21475541` (failed closed because the candidate was incomplete)

Calibration and paper-campaign submission remain unauthorized for V5.
