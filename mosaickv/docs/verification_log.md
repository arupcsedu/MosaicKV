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
- Second synchronization attempt: all locked packages downloaded and installed,
  then both `pip check` and `uv pip check` rejected `decord==0.6.0` because its
  installed wheel is tagged `cp36-cp36m-manylinux2010_x86_64`, not CPython
  3.11. Imports and support remained unverified at this boundary.
- Corrective action: do not waive the checker. Replace SGLang's legacy video
  extra with an explicit SRT dependency set and `decord2==3.4.0`, whose wheel is
  tagged `cp311-cp311-manylinux_2_28_x86_64` and provides SGLang's imported
  `decord` module API.
- Third synchronization attempt: exact package synchronization and `pip check`
  passed. The unbounded import verifier then made no progress for about 49
  minutes on the login node and was terminated. No import or backend support
  claim was made from that attempt.
- Slurm environment smoke job `17181271` at clean commit `aef4457` ran on one
  NVIDIA A100-SXM4-80GB. All 243 pins matched, cache paths resolved under
  `/scratch/djy8hg/cache/mosaickv`, and the CUDA 12.4 matrix multiplication
  passed. The job failed because SGLang 0.4.3.post1 imported
  `is_valid_list_of_images` from a Transformers 4.49.0 module that no longer
  exports it.
- Corrective action: pin SGLang 0.4.3.post4, whose upstream source defines the
  compatibility helper locally, and isolate each verifier import behind a
  hard deadline. Environment creation no longer starts heavyweight imports on
  the login node; the clean-tree Slurm smoke owns that verification gate.
- Environment synchronization job `17182039` at clean commit `8d45645`
  completed in five seconds. It installed SGLang 0.4.3.post4 and `pip check`
  reported no broken requirements.
- Bounded Slurm smoke job `17182181` completed rather than hanging. All module
  imports were individually bounded and the CUDA matrix multiplication passed,
  but support remained failed for two reasons: SGLang's compatibility config
  attempted to register a processor class name already native to Transformers
  4.49, and `sgl_kernel` could not locate the wheel-provided `libnvrtc.so.12`.
- Corrective action: add a fail-closed, versioned SGLang registration patch
  using the public `exist_ok=True` parameter, verify and manifest its SHA, and
  expose the locked `nvidia-cuda-nvrtc-cu12` library directory through the
  common cache/environment bootstrap. No model code or inference math is
  changed by either compatibility fix.
