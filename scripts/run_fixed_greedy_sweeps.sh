#!/usr/bin/env bash
set -uo pipefail

# Interaction-aware selection gate — corrected greedy rerun.
#
# Runs on every model that HAS a discovered LLMC GPTQ fake-quant checkpoint and
# WARNS-AND-SKIPS any that does not (e.g. 8B), instead of aborting the whole run
# (the previous preflight blocked all sweeps on the missing 8B checkpoint).
#
# For each available model it evaluates BOTH selection modes on BOTH bases:
#   --select greedy        interaction-aware, iterative RX update
#   --select greedy_indep  same objective's first-step gains, NO iterative update
# so build_comparison can compare them, at matched actual weight bits, against the
# existing residual_max / residual_rms / act_max / random rows and answer the
# decisive question: do the iterative cross-column interactions materially help?
#
# It writes to fresh out_dirs and never overwrites the historical invalid greedy
# outputs. Override the repo root with SEQ_ROOT if needed.
ROOT="${SEQ_ROOT:-/mnt/d/Abrar/SEQ/seq_v4}"
OUT_ROOT="$ROOT/runs/final_greedy_fixed"
LOG="$OUT_ROOT/run.log"
CALIB="$ROOT/calibration_prompts.json"
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

SELECTS=( greedy greedy_indep )
FRACS="0,0.02,0.05,0.1,0.2"

run_sweep () {  # model out_dir extra_args...
  local model="$1"; local out="$2"; shift 2
  echo "$(date -Is) RUN $model -> $out :: $*"
  python -m seq_core.channel_sweep \
    --model "$model" --backend hqq --base_bits 4 \
    --protect_fracs "$FRACS" \
    --ppl_mode canonical --calibration_prompts "$CALIB" \
    --out_dir "$out" "$@" \
    || echo "$(date -Is) SWEEP FAILED (continuing): $model -> $out"
}

{
  echo "$(date -Is) interaction-aware greedy gate — preflight"
  test -r "$CALIB" || { echo "MISSING calibration_prompts.json: $CALIB"; exit 3; }

  AVAILABLE=()
  for MODEL in "${MODELS[@]}"; do
    P="${GPTQ_PATHS[$MODEL]+${GPTQ_PATHS[$MODEL]}}"
    if [[ -n "${P:-}" && -f "$P/config.json" \
          && ( -f "$P/model.safetensors" || -f "$P/model.safetensors.index.json" ) ]]; then
      AVAILABLE+=("$MODEL"); echo "OK   $MODEL -> $P"
    else
      echo "SKIP $MODEL (no valid GPTQ fake-quant checkpoint${P:+ at $P})"
    fi
  done
  [[ ${#AVAILABLE[@]} -gt 0 ]] || { echo "no models available; nothing to do"; exit 0; }
  echo "launching gate sweeps for: ${AVAILABLE[*]}"
} 2>&1 | tee "$LOG"

for MODEL in "${AVAILABLE[@]}"; do
  NAME="${MODEL##*/}"
  GP="${GPTQ_PATHS[$MODEL]}"
  for SEL in "${SELECTS[@]}"; do
    # GPTQ base (the decisive one) and HQQ base (data-free; the priority bug
    # invalidated HQQ greedy too, so it is re-run here as well).
    run_sweep "$MODEL" "$OUT_ROOT/gptq_${NAME}_${SEL}" \
      --select "$SEL" --base_quantizer gptq_llmc --gptq_model_path "$GP" 2>&1 | tee -a "$LOG"
    run_sweep "$MODEL" "$OUT_ROOT/hqq_${NAME}_${SEL}" \
      --select "$SEL" 2>&1 | tee -a "$LOG"
  done
done

echo "$(date -Is) gate sweeps complete" | tee -a "$LOG"
echo "Next (CPU): python analysis/build_comparison.py --sweeps runs/final_* \\" | tee -a "$LOG"
echo "  --baselines <baselines.json> \\" | tee -a "$LOG"
echo "  --signals greedy,greedy_indep,residual_max,residual_rms,act_max,random \\" | tee -a "$LOG"
echo "  --out docs/COMPARISON.md" | tee -a "$LOG"
