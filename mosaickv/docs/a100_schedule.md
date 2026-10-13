# A100 allocation rule (updated 2026-10-13)

**At most four available A100s at all times**, in aggregate across MosaicKV
jobs, not four per array. This supersedes the earlier eight-GPU overnight
allowance. Midnight and daylight-saving changes do not change the limit.
Running jobs are never canceled or suspended to meet the cap: if occupancy
exceeds four, no additional worker is admitted.

Every GPU task uses `bii-gpu`, reservation `bi_fox_dgx`, one A100, 12 CPUs,
and a two-hour maximum. Baseline workers keep their existing 180-GiB host-RAM
request and immutable experimental configurations. Other active A100 jobs
owned by this user count against the ceiling without being modified.

## Admission mechanism

`scripts/control_a100_schedule.py` controls registered arrays using pending-only
holds/releases. Released-but-pending tasks reserve slots, preventing separate
array throttles from multiplying the limit. Each managed array also has
`ArrayTaskThrottle=4`; the controller rejects admission limits above four.

`JobHeldUser` is intentional admission control, not a broken dependency.
Do not manually release all tasks from multiple arrays. Register future arrays
with the controller before release. No other user's jobs are modified.

The one-CPU, 256-MiB controller runs on `bii`, checks every 30 seconds, and exits
when its arrays finish. A filesystem lock prevents duplicate controllers.
Queue/control errors fail closed. Code, policy, state, caches, and logs remain
under scratch. An administrative snapshot does not modify experiment code or
require a source commit.

## Current deployment

Snapshot: `/scratch/djy8hg/runs/mosaickv/a100-repair-20261013-v1/`.

| Job | Purpose | Admission |
| --- | --- | --- |
| `21292854_2–255` | Remaining HF shared-byte baseline shards | Four total across arrays |
| `21314743_0–1` | Retry two shards that failed before inference | Same global allowance |
| `21314746` | CPU admission controller | Earliest start October 15, 06:00 EDT |

The obsolete pending controller `21302584` was canceled and replaced so that
its old eight-GPU policy cannot return. No running GPU jobs were canceled.
Four tasks were released initially: `21292854_2`, `21292854_3`,
`21314743_0`, and `21314743_1`. All others remain held until slots are available.

Cluster-wide maintenance covers **October 13, 06:00 through October 15, 06:00
EDT**, including `udc-an26-1`. The configured `bi_fox_dgx` reservation cannot
bypass maintenance. The replacement controller is scheduled for maintenance
end; allocation at exactly that time is not guaranteed.

## Missing PrefixKV input repaired

The original failures were caused by an uninitialized git submodule.
Initialized only the existing `third_party/PrefixKV` gitlink in the isolated
experiment worktree:

- MosaicKV remains clean at `2c7a266d05a0dd30a4a7440355148549c14baeea`.
- PrefixKV is pinned at `597f1ab032704951550f93bcc8a23f1454b80aa4`.
- Restored profile: `confs/prefixkv_llava-v1.5-7b_0.8.json`.
- Profile SHA-256: `0797d4359f7e9ded349808be00c2d29b760f658df8c4d002ad6ca5824e2b9d67`.
- File contents match the pinned commit byte-for-byte; its MIT license is
  preserved. No upstream source was patched and no package was installed.

`deployment_validation.json` in the snapshot records CPU verification of all
256 job-spec hashes, 64 sample-ledger hashes, 16 unique configuration groups,
the profile loader, and shared compressed-method controls. This is a deployment
check, not a new GPU quality/latency result. Failed logs are preserved; retries
use new Slurm IDs. Neither failed shard had inference artifacts to overwrite.
All completed HF and vLLM results are untouched.

Verification: 19 CPU scheduler tests and Ruff pass, covering the constant cap,
DST, aggregate accounting, pending reservations, hold ownership, retries, and
throttle readback after completed sibling jobs.
