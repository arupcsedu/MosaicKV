#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "${PROJECT_ROOT}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  printf 'Refusing a canonical run: tracked worktree changes are present.\n' >&2
  git status --short >&2
  exit 2
fi

untracked=$(git ls-files --others --exclude-standard)
if [[ -n "${untracked}" ]]; then
  printf 'Refusing a canonical run: untracked files are present.\n%s\n' "${untracked}" >&2
  exit 2
fi

printf 'clean_git_sha=%s\n' "$(git rev-parse HEAD)"
