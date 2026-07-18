#!/usr/bin/env bash
set -euo pipefail
MODELS="meta-llama/Llama-3.2-1B,meta-llama/Llama-3.2-3B"
LLMC_REPO=""; LLMC_VENV=""; OUTPUT_ROOT="results/final_baselines"; DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2;;
    --llmc-repo) LLMC_REPO="$2"; shift 2;;
    --llmc-venv) LLMC_VENV="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; RUN_DIR="$OUTPUT_ROOT/$STAMP"; mkdir -p "$RUN_DIR"
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
for model in "${MODEL_ARR[@]}"; do
  slug="${model//\//_}"; dir="$RUN_DIR/$slug"
  CMD=(python scripts/run_llmc_w4_baselines.py --model "$model" --methods rtn,awq,gptq --out-dir "$dir")
  [[ -n "$LLMC_REPO" ]] && CMD+=(--llmc-repo "$LLMC_REPO")
  [[ -n "$LLMC_VENV" ]] && CMD+=(--llmc-venv "$LLMC_VENV")
  printf 'baseline command:'; printf ' %q' "${CMD[@]}"; echo
  if [[ "$DRY_RUN" == 0 ]]; then "${CMD[@]}"; fi
done
python scripts/index_baseline_results.py --root "$RUN_DIR" --output "$OUTPUT_ROOT/model_index.json" --models "$MODELS"
