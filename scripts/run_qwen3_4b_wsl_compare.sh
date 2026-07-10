#!/usr/bin/env bash
set -euo pipefail

SEQ_ROOT="${SEQ_ROOT:-/mnt/e/SEQ_Clean}"
LLMC_REPO="${LLMC_REPO:-/mnt/e/LightCompress}"
LLMC_VENV="${LLMC_VENV:-$LLMC_REPO/.venv-llmc}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
METHODS="${METHODS:-base,seq,gptq_llmc,smoothquant_llmc,awq_llmc,rtn_llmc,omniquant_llmc}"
BENCHMARKS="${BENCHMARKS:-ppl,hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai}"
LLMC_MODEL_TYPE="${LLMC_MODEL_TYPE:-Qwen2}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${TS}}"

[[ -d "$SEQ_ROOT" ]] || { echo "SEQ repo not found: $SEQ_ROOT" >&2; exit 1; }
[[ -f "$LLMC_VENV/bin/activate" ]] || { echo "LLMC venv not found: $LLMC_VENV/bin/activate" >&2; exit 1; }
[[ "$METHODS" == *omniquant_llmc* ]] && echo "Note: omniquant_llmc is experimental/not recommended in validation notes." >&2

cd "$SEQ_ROOT"
# shellcheck disable=SC1091
source "$LLMC_VENV/bin/activate"
export PYTHONPATH="$LLMC_REPO:${PYTHONPATH:-}"

args=(
  run_compare_matrix.py
  --models "$MODEL"
  --device "${DEVICE:-auto}"
  --methods "$METHODS"
  --benchmarks "$BENCHMARKS"
  --experiments_file "${EXPERIMENTS_FILE:-experiments.yaml}"
  --output_dir "$OUTPUT_DIR"
  --lm_eval_num_fewshot "${LM_EVAL_NUM_FEWSHOT:-0}"
  --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE:-1}"
  --lm_eval_fail_policy "${LM_EVAL_FAIL_POLICY:-warn}"
  --llmc_repo "$LLMC_REPO"
  --llmc_venv "$LLMC_VENV"
  --llmc_model_type "$LLMC_MODEL_TYPE"
  --llmc_save_mode "${LLMC_SAVE_MODE:-fake}"
  --llmc_calib_dataset "${LLMC_CALIB_DATASET:-wikitext2}"
  --llmc_eval_dataset "${LLMC_EVAL_DATASET:-wikitext2}"
  --llmc_calib_samples "${LLMC_CALIB_SAMPLES:-32}"
  --llmc_calib_seq_len "${LLMC_CALIB_SEQ_LEN:-512}"
  --llmc_eval_seq_len "${LLMC_EVAL_SEQ_LEN:-2048}"
)
[[ -n "${LM_EVAL_LIMIT:-}" ]] && args+=(--lm_eval_limit "$LM_EVAL_LIMIT")
[[ "${LLMC_DRY_RUN:-0}" == "1" ]] && args+=(--llmc_dry_run)
printf 'Running: python'; printf ' %q' "${args[@]}"; printf '\n'
python "${args[@]}"
