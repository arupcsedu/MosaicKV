# V6 surrogate timing diagnostic seal

V6 is sealed as a negative development timing result. It is not paper evidence,
and its 16-sample canary and 128-sample calibration were not submitted.

- Source base SHA: `8566a826578d5e4e3b4b6bec700b3fb7e5389d68`
- Plan SHA-256: `f76907e1a735942cdcd587d9e3a18dd018f332012dc28a8774b283f5b912fd1d`
- Worker SHA-256: `3c6a231a2e843dc5070a720ec30936c7e05f22155f8366abff452765491bacd7`
- Analysis SHA-256: `5e7bd2376f3e24474726cff71cf68afaf943c95b22b623979d936c98824e1521`
- Slurm worker: `21499283_0`

The run passed retention-one parity with 100% token agreement and zero maximum
logit difference. It retained exactly 25.5 MiB initially, used zero padding and
zero residual storage, and emitted complete timing records. However, candidate
generation took 2,493.4806 seconds and worker wall time was 2,802.5047 seconds,
exceeding the preregistered five-second generation and 360-second worker gates.
Both two-token candidates exceeded the fixed KL/logit thresholds, so the method
failed closed to its Prompt Top-K anchor. No benchmark labels were accessed.

This result authorizes neither a quality claim nor successor submission.
