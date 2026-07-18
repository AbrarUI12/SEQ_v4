#!/usr/bin/env bash
set -euo pipefail
MODELS="meta-llama/Llama-3.2-1B,meta-llama/Llama-3.2-3B"; OUT="runs/hqq_uniform"; PROMPTS="calibration_prompts.json"; DEVICE="cuda"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2;;
    --output-root) OUT="$2"; shift 2;;
    --calibration-prompts) PROMPTS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
for model in "${MODEL_ARR[@]}"; do
  slug="${model//\//_}"
  for bits in 4 5 6 8; do
    dir="$OUT/$slug/b${bits}"; mkdir -p "$dir"
    cmd=(python -m seq_core.channel_sweep --model "$model" --backend hqq --base_bits "$bits" --protect_fracs 0 --signals act_max --ppl_mode canonical --calibration_prompts "$PROMPTS" --out_dir "$dir" --device "$DEVICE")
    printf '%q ' "${cmd[@]}"; echo
    [[ "$DRY" == 0 ]] && "${cmd[@]}"
  done
done
