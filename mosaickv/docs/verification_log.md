# Environment and container verification log

This log preserves setup and verification outcomes. It is not a measured-results
table and does not establish model or backend support.

## 2026-07-21 common environment reset

- Source commit: `f2181df21dc735f710c9539354ecd123de0bdc55`
- Cache root: `/scratch/djy8hg/cache/mosaickv`
- Prefix: `/scratch/djy8hg/env/mosaickv`
- Lock resolution: passed with uv for 243 exact distributions on CPython 3.11,
  Linux x86_64/manylinux 2.28.
- First synchronization attempt: failed while downloading/extracting
  `opencv-python-headless==4.11.0.86`; uv reported a network timeout at its
  default 30-second HTTP timeout.
- Interpretation: dependency resolution passed; installation, imports, CUDA,
  backend support, and Docker remained unverified after this attempt.
- Corrective action: version a 300-second uv HTTP timeout, ten retries, and
  four concurrent downloads for host and container setup, then resume from the
  scratch cache.
