#!/usr/bin/env bash
set -euo pipefail
MODELS="meta-llama/Llama-3.2-1B,meta-llama/Llama-3.2-3B"; LLMC_REPO=""; LLMC_VENV=""; OUT="runs/final"; PHASE="all"; RESUME=0; DRY=0; FAIL_FAST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2;; --llmc-repo) LLMC_REPO="$2"; shift 2;; --llmc-venv) LLMC_VENV="$2"; shift 2;;
    --output-root) OUT="$2"; shift 2;; --phase) PHASE="$2"; shift 2;; --resume) RESUME=1; shift;;
    --dry-run) DRY=1; shift;; --fail-fast) FAIL_FAST=1; shift;; --skip-8b) shift;; *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
mkdir -p "$OUT"; failures=0
run_phase() {
  local name="$1"; shift; local marker="$OUT/.${name}.complete"
  [[ "$PHASE" != all && "$PHASE" != "$name" ]] && return 0
  [[ "$RESUME" == 1 && -f "$marker" ]] && { echo "skip $name (marker exists)"; return 0; }
  echo "=== phase: $name ==="; printf 'command:'; printf ' %q' "$@"; echo
  if [[ "$DRY" == 1 ]]; then return 0; fi
  if "$@"; then date -u > "$marker"; else failures=$((failures+1)); [[ "$FAIL_FAST" == 1 ]] && exit 1; fi
}
run_phase environment python scripts/audit_final_environment.py --output "$OUT/environment.json"
run_phase tests python -m compileall seq_core analysis scripts
run_phase tests_pytest python -m pytest -q
run_phase baselines bash scripts/run_final_baselines.sh --models "$MODELS" --llmc-repo "$LLMC_REPO" --llmc-venv "$LLMC_VENV" --output-root "$OUT/baselines"
run_phase hqq_uniform bash scripts/run_uniform_hqq_sweep.sh --models "$MODELS" --output-root "$OUT/hqq_uniform"
run_phase hqq_residual python -m seq_core.channel_sweep --model "${MODELS%%,*}" --backend hqq --base_bits 4 --protect_fracs 0,0.02,0.05,0.1,0.2 --signals act_max,residual_rms,residual_max,random --ppl_mode canonical --calibration_prompts calibration_prompts.json --out_dir "$OUT/seq10_residual"
run_phase hqq_greedy python -m seq_core.channel_sweep --model "${MODELS%%,*}" --backend hqq --base_bits 4 --protect_fracs 0,0.02,0.05,0.1,0.2 --select greedy --ppl_mode canonical --calibration_prompts calibration_prompts.json --out_dir "$OUT/seq10_greedy"
run_phase validate python analysis/validate_final_results.py "$OUT"
run_phase comparison python analysis/build_comparison.py --sweeps "$OUT" --baselines baselines.json --signals act_max,residual_rms,residual_max,greedy,random,act_scale --out docs/COMPARISON.md
run_phase plots python analysis/plot_final_results.py --root "$OUT" --output-dir figures
echo "pipeline complete; failures=$failures"; exit "$failures"
