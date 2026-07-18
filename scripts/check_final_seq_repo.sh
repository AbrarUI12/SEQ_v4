#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
status=0
git status || status=$?
python scripts/audit_final_environment.py --output results/final_environment_audit.json || status=$?
python -m compileall seq_core analysis scripts || status=$?
python -m pytest -q || status=$?
exit "$status"
