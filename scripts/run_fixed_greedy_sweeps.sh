#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MATRIX="$ROOT/configs/final_comparison_matrix.json"
OUT_ROOT="$ROOT/runs/final/sweeps"
GPTQ_ROOT="$ROOT/runs/final/llmc"
MODELS_OVERRIDE=""
DEVICE="auto"
RESUME=0
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix) MATRIX="$2"; shift 2 ;;
    --output-root) OUT_ROOT="$2"; shift 2 ;;
    --gptq-root) GPTQ_ROOT="$2"; shift 2 ;;
    --models) MODELS_OVERRIDE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PYTHON="$ROOT/.venv-seq/bin/python"
[[ "$DRY" == 1 || -x "$PYTHON" ]] || { echo "SEQ Python not found: $PYTHON" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "matrix not found: $MATRIX" >&2; exit 2; }
cd -- "$ROOT"
if [[ -n "$MODELS_OVERRIDE" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_OVERRIDE"
else
  mapfile -t MODELS < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["models"]))' "$MATRIX")
fi
mapfile -t SELECTORS < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["set_selectors"]))' "$MATRIX")
mapfile -t BASE_SPECS < <("$PYTHON" -c 'import json,sys; print("\n".join("{}|{}".format(x["name"],x["base_bits"]) for x in json.load(open(sys.argv[1]))["bases"]))' "$MATRIX")
FRACS="$("$PYTHON" -c 'import json,sys; print(",".join(str(x) for x in json.load(open(sys.argv[1]))["fractions"] if float(x)!=0))' "$MATRIX")"
SEED="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["deterministic_seed"])' "$MATRIX")"
CALIB="$ROOT/calibration_prompts.json"
mkdir -p "$OUT_ROOT"

run_one() {
  local model="$1" base="$2" base_bits="$3" selector="$4" checkpoint="$5"
  local name="${model##*/}"
  local out="$OUT_ROOT/$base/$name/$selector/seed-$SEED"
  if [[ "$RESUME" == 1 && -s "$out/channel_pareto.json" ]]; then
    echo "reuse existing sweep: $out/channel_pareto.json"
    return 0
  fi
  local cmd=("$PYTHON" -m seq_core.channel_sweep --model "$model" --backend hqq --base_bits "$base_bits"
             --base_quantizer "$base" --protect_fracs "$FRACS" --select "$selector" --seed "$SEED"
             --device "$DEVICE" --ppl_mode canonical --calibration_prompts "$CALIB" --out_dir "$out")
  [[ "$base" == gptq_llmc ]] && cmd+=(--gptq_model_path "$checkpoint")
  printf 'command:'; printf ' %q' "${cmd[@]}"; echo
  [[ "$DRY" == 1 ]] || "${cmd[@]}"
}

for model in "${MODELS[@]}"; do
  checkpoint="$GPTQ_ROOT/${model##*/}/gptq/artifacts/fake_quant_model"
  if [[ "$DRY" != 1 ]]; then
    [[ -f "$checkpoint/config.json" ]] || { echo "missing GPTQ checkpoint config: $checkpoint" >&2; exit 3; }
    [[ -f "$checkpoint/model.safetensors" || -f "$checkpoint/model.safetensors.index.json" ]] || {
      echo "missing GPTQ checkpoint weights: $checkpoint" >&2; exit 3;
    }
  fi
  for spec in "${BASE_SPECS[@]}"; do
    IFS='|' read -r base base_bits <<< "$spec"
    for selector in "${SELECTORS[@]}"; do
      run_one "$model" "$base" "$base_bits" "$selector" "$checkpoint"
    done
  done
done

echo "greedy gate sweeps complete"
