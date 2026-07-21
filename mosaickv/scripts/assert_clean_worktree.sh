#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

if ! git -C "${REPOSITORY_ROOT}" rev-parse --is-inside-work-tree >/dev/null; then
  printf 'Unable to resolve the MosaicKV repository at %s\n' "${REPOSITORY_ROOT}" >&2
  exit 2
fi

if ! git -C "${REPOSITORY_ROOT}" diff --quiet || \
  ! git -C "${REPOSITORY_ROOT}" diff --cached --quiet; then
  printf 'Refusing a canonical run: tracked worktree changes are present.\n' >&2
  git -C "${REPOSITORY_ROOT}" status --short >&2
  exit 2
fi

untracked=$(git -C "${REPOSITORY_ROOT}" ls-files --others --exclude-standard)
if [[ -n "${untracked}" ]]; then
  printf 'Refusing a canonical run: untracked files are present.\n%s\n' "${untracked}" >&2
  exit 2
fi

printf 'clean_git_sha=%s\n' "$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)"
