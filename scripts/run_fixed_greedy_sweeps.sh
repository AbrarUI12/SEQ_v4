#!/usr/bin/env bash
set -euo pipefail

# Corrected greedy rerun.  The preflight intentionally fails if any required
# model lacks a discovered LLMC GPTQ fake-quant checkpoint; it never guesses a
# path and never overwrites the historical invalid greedy outputs.
ROOT="/mnt/d/Abrar/SEQ/seq_v4"
OUT_ROOT="$ROOT/runs/final_greedy_fixed"
LOG="$OUT_ROOT/run.log"
source "$ROOT/.venv-seq/bin/activate"
mkdir -p "$OUT_ROOT"

declare -A GPTQ_PATHS=(
  ["meta-llama/Llama-3.2-3B"]="$ROOT/runs/final_llmc/Llama-3.2-3B/gptq/artifacts/fake_quant_model"
  ["meta-llama/Llama-3.2-1B"]="$ROOT/runs/final_llmc/Llama-3.2-1B/gptq/artifacts/fake_quant_model"
)

MODELS=(
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.2-3B"
  "meta-llama/Llama-3.2-1B"
)

{
  echo "$(date -Is) fixed greedy preflight"
  for MODEL in "${MODELS[@]}"; do
    if [[ -z "${GPTQ_PATHS[$MODEL]+present}" ]]; then
      echo "MISSING GPTQ CHECKPOINT MAPPING: $MODEL"
      exit 2
    fi
    GPTQ_MODEL_PATH="${GPTQ_PATHS[$MODEL]}"
    if [[ ! -d "$GPTQ_MODEL_PATH" || ! -f "$GPTQ_MODEL_PATH/config.json" || ! -f "$GPTQ_MODEL_PATH/model.safetensors" ]]; then
      echo "MISSING OR INVALID GPTQ CHECKPOINT: $MODEL -> $GPTQ_MODEL_PATH"
      exit 2
    fi
  done
  test -r "$ROOT/calibration_prompts.json"
  echo "preflight passed; launching corrected sweeps"
} 2>&1 | tee "$LOG"

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME="${MODEL##*/}"
  GPTQ_MODEL_PATH="${GPTQ_PATHS[$MODEL]}"
  python -m seq_core.channel_sweep \
    --model "$MODEL" \
    --backend hqq \
    --base_bits 4 \
    --select greedy \
    --protect_fracs 0,0.02,0.05,0.1,0.2 \
    --base_quantizer gptq_llmc \
    --gptq_model_path "$GPTQ_MODEL_PATH" \
    --ppl_mode canonical \
    --calibration_prompts "$ROOT/calibration_prompts.json" \
    --out_dir "$OUT_ROOT/$MODEL_NAME" 2>&1 | tee -a "$LOG"
done
