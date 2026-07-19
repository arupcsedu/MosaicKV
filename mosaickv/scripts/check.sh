#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

"${PYTHON_BIN}" -m ruff check src tests scripts/verify_envs.py
"${PYTHON_BIN}" -m ruff format --check src tests scripts/verify_envs.py
"${PYTHON_BIN}" -m mypy src tests scripts/verify_envs.py
"${PYTHON_BIN}" -m pytest -q
