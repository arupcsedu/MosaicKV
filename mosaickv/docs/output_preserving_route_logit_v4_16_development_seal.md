# V4 route-logit canary seal

V4 is sealed as a negative, development-only result. Its 16 identities are
consumed and must not appear in calibration, powered validation, or paper
evidence. No sample-level answer, reference, score, or failure identity from
this cohort may be used to tune a successor.

The completed canary passed retention-one parity, exact byte accounting,
state-isolation, provenance, trace completeness, and the memory gate. It did
not pass the preregistered quality-confidence or fallback gates:

- minimum initial/final/peak reduction versus Prompt Top-K at 40 MiB:
  `30.287206266318534%`;
- candidate minus Prompt Top-K CIDEr: `+2.1748657015357225`;
- paired 95% interval: `[-2.647412179884909, 8.494938405901582]`;
- fail-closed fallback: `10/16` (`62.5%`);
- the V4 answers exactly matched the 25.5 MiB Prompt Top-K anchor.

The aggregate label-blind audit also found that V4 composed same-cardinality
within-route substitutions. It did not transfer physical capacity from a
low-impact donor route to a high-impact receiver route. A successor must use
a separately versioned design and new identities.

## Immutable artifact bindings

- plan: `22ea1378ebc49209d582947029100d3fd4527c5ae9bc6adf141ca09241d484ad`
- ledger: `9ab24cee05f3621fc12aaa1e51dd7916c05388512e75603fb3cb6c0ed913fef0`
- report: `4386f80af48efab7a97cdc817ea3a486c3f85157829268afcfb6fbfb4de9f107`
- paired samples: `e99dc458d8f50a0d7b739b5ca06c97329a8fec09902de0593ba6b040f91c0ba5`
- quality comparisons: `1204f4710a7a3472dbb7e789db654c3786c29f3cbcec0e82cfe17a667873a040`
- byte gates: `c71a2d5c8df05c0ecb53bbb5e1754dfa1f8733488fc56c7d60ecf3fdc716e9e8`

Calibration and paper-campaign submission remain unauthorized for V4.
