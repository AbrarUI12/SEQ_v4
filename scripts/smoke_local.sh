#!/usr/bin/env bash
# smoke_local.sh — the gate before code goes to the research PC (5090/4090).
#
# Two phases, auto-detected:
#   CPU phase  (always): py_compile + pure-stdlib tests. Runs anywhere (even the
#              Windows Python with no torch/pytest).
#   GPU phase  (when torch imports): torch-dependent tests + a 1B channel_sweep
#              micro-smoke exercising --verify_materialized + a downstream dry-run.
#
# Usage:  bash scripts/smoke_local.sh            # from repo root
#         PY=python3 bash scripts/smoke_local.sh # override interpreter
#
# Exit non-zero if any run in the active phase fails.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

fail=0
run() {  # run <label> <cmd...>
  local label="$1"; shift
  printf '\n=== %s ===\n' "$label"
  if "$@"; then printf '  [PASS] %s\n' "$label"
  else printf '  [FAIL] %s\n' "$label"; fail=$((fail+1)); fi
}

have() { "$PY" -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }

HAS_TORCH=0;  have torch  && HAS_TORCH=1
HAS_PYTEST=0; have pytest && HAS_PYTEST=1
echo "interpreter=$($PY --version 2>&1)  torch=$HAS_TORCH  pytest=$HAS_PYTEST"

# ---- CPU phase ------------------------------------------------------------- #
run "py_compile (seq_core, scripts, analysis)" \
  bash -c "$PY -m py_compile seq_core/*.py analysis/*.py 2>&1 && echo compiled"

# self-running, no-torch tests (exit non-zero on failure via their __main__)
for t in test_stats_utils test_channel_utils test_comparison test_gptq_math; do
  run "tests/$t.py" "$PY" "tests/$t.py"
done

# bare-pytest, no-torch tests — only if pytest is present
NOTORCH_PYTEST="tests/test_storage_accounting.py tests/test_final_matrix.py \
tests/test_final_portability.py tests/test_llmc_w4_baselines.py tests/test_publish_final_results.py"
if [ "$HAS_PYTEST" = 1 ]; then
  run "pytest (no-torch suite)" "$PY" -m pytest -q $NOTORCH_PYTEST
else
  echo -e "\n(skip) pytest not installed — bare-func no-torch tests not run: $NOTORCH_PYTEST"
fi

# ---- GPU phase (only when torch is importable) ----------------------------- #
if [ "$HAS_TORCH" = 1 ]; then
  # torch-dependent tests
  run "tests/test_greedy_select.py" "$PY" tests/test_greedy_select.py
  if [ "$HAS_PYTEST" = 1 ]; then
    run "pytest (torch suite)" "$PY" -m pytest -q tests/test_channel_export.py tests/test_gptq_sequential.py
  fi
  # 1B micro-smoke: exercises --verify_materialized end-to-end (tiny ppl budget)
  run "channel_sweep 1B micro-smoke (--verify_materialized)" \
    "$PY" -m seq_core.channel_sweep --model meta-llama/Llama-3.2-1B --backend hqq \
      --base_bits 4 --protect_fracs 0.02 --select greedy --seed 1234 \
      --ppl_mode canonical --ppl_max_examples 8 \
      --calibration_prompts calibration_prompts.json \
      --out_dir /tmp/seq_smoke --verify_materialized
  # pipeline command-construction check (no GPU cost)
  run "run_downstream_eval.sh --dry-run" bash scripts/run_downstream_eval.sh --dry-run
else
  echo -e "\n(skip) torch not importable — GPU phase skipped (run this on the WSL .venv-seq)"
fi

# ---- summary --------------------------------------------------------------- #
printf '\n============================================\n'
if [ "$fail" -eq 0 ]; then echo "SMOKE OK (phase: $([ $HAS_TORCH = 1 ] && echo GPU || echo CPU-only))"; exit 0
else echo "SMOKE FAILED: $fail run(s) failed"; exit 1; fi
