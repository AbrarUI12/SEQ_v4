#!/usr/bin/env bash
set -euo pipefail

LLMC_REPO="${1:-/mnt/e/LightCompress}"
METHODS_RAW="${2:-gptq,smoothquant,awq,rtn,llm_int8}"
SEQ_ROOT="/mnt/e/SEQ_Clean"
LOG_DIR="$SEQ_ROOT/results/llmc_smoke/logs"

if [[ ! -d "$LLMC_REPO" ]]; then
  echo "LLMC repo not found: $LLMC_REPO" >&2
  exit 1
fi

if [[ ! -f "$LLMC_REPO/.venv-llmc/bin/activate" ]]; then
  echo "LLMC venv not found: $LLMC_REPO/.venv-llmc/bin/activate" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

source "$LLMC_REPO/.venv-llmc/bin/activate"
export PYTHONPATH="$LLMC_REPO:${PYTHONPATH:-}"

run_method() {
  local method_name="$1"
  local output_dir="$2"
  local config_path="$3"
  local task_id="$4"
  local log_path="$5"

  if [[ -d "$output_dir" ]]; then
    rm -rf "$output_dir"
  fi

  echo "=== $method_name memory snapshot ==="
  free -h
  nvidia-smi

  (
    cd "$LLMC_REPO"
    torchrun --standalone --nproc_per_node=1 llmc/__main__.py \
      --config "$config_path" \
      --task_id "$task_id"
  ) 2>&1 | tee "$log_path"
}

IFS=',' read -r -a METHODS <<< "$METHODS_RAW"
for METHOD in "${METHODS[@]}"; do
  case "$METHOD" in
    gptq)
      run_method \
        "GPTQ" \
        "$SEQ_ROOT/results/llmc_smoke/gptq_opt125m_tiny_128" \
        "$SEQ_ROOT/third_party_quant/llmc_smoke_configs/gptq_opt125m_smoke.yml" \
        "gptq_opt125m_tiny_128" \
        "$LOG_DIR/gptq_opt125m_tiny_128.log"
      ;;
    smoothquant)
      run_method \
        "SmoothQuant" \
        "$SEQ_ROOT/results/llmc_smoke/smoothquant_opt125m_tiny_128" \
        "$SEQ_ROOT/third_party_quant/llmc_smoke_configs/smoothquant_opt125m_smoke.yml" \
        "smoothquant_opt125m_tiny_128" \
        "$LOG_DIR/smoothquant_opt125m_tiny_128.log"
      ;;
    awq)
      run_method \
        "AWQ" \
        "$SEQ_ROOT/results/llmc_smoke/awq_opt125m_tiny_128" \
        "$SEQ_ROOT/third_party_quant/llmc_smoke_configs/awq_opt125m_smoke.yml" \
        "awq_opt125m_tiny_128" \
        "$LOG_DIR/awq_opt125m_tiny_128.log"
      ;;
    rtn)
      run_method \
        "RTN" \
        "$SEQ_ROOT/results/llmc_smoke/rtn_opt125m_tiny_128" \
        "$SEQ_ROOT/third_party_quant/llmc_smoke_configs/rtn_opt125m_smoke.yml" \
        "rtn_opt125m_tiny_128" \
        "$LOG_DIR/rtn_opt125m_tiny_128.log"
      ;;
    llm_int8)
      run_method \
        "LlmInt8" \
        "$SEQ_ROOT/results/llmc_smoke/llm_int8_opt125m_tiny_128" \
        "$SEQ_ROOT/third_party_quant/llmc_smoke_configs/llm_int8_opt125m_smoke.yml" \
        "llm_int8_opt125m_tiny_128" \
        "$LOG_DIR/llm_int8_opt125m_tiny_128.log"
      ;;
    spinquant)
      echo "SpinQuant smoke is not available in the current LLMC checkout: no upstream SpinQuant config or method was found, and the nearest QuaRot path requires fast_hadamard_transform." >&2
      ;;
    omniquant)
      echo "OmniQuant smoke is intentionally disabled; see third_party_quant/docs/llmc_validation.md." >&2
      ;;
    *)
      echo "Unsupported smoke method: $METHOD" >&2
      exit 1
      ;;
  esac
done
