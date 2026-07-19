#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=""; TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai"; OUT="results/final_seq_tasks"; DEVICE="cuda:0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2;;
    --tasks) TASKS="$2"; shift 2;;
    --output-dir) OUT="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$MODEL_PATH" ]] || { echo '--model-path is required' >&2; exit 2; }
python -m lm_eval --model hf --model_args "pretrained=$MODEL_PATH" --tasks "$TASKS" --num_fewshot 0 --batch_size 1 --device "$DEVICE" --output_path "$OUT"
