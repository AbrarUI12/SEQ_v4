#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MATRIX="$ROOT/configs/final_comparison_matrix.json"
MODELS_OVERRIDE=""
LLMC_REPO=""
LLMC_VENV=""
OUT="$ROOT/runs/final"
PHASE="all"
RESUME=0
DRY=0
FAIL_FAST=1
PUBLISH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS_OVERRIDE="$2"; shift 2 ;;
    --matrix) MATRIX="$2"; shift 2 ;;
    --llmc-repo) LLMC_REPO="$2"; shift 2 ;;
    --llmc-venv) LLMC_VENV="$2"; shift 2 ;;
    --output-root) OUT="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --publish) PUBLISH=1; shift ;;
    --skip-8b) shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$LLMC_REPO" ]] || { echo "--llmc-repo is required" >&2; exit 2; }
[[ -n "$LLMC_VENV" ]] || { echo "--llmc-venv is required" >&2; exit 2; }
[[ "$LLMC_REPO" != *[[:space:]]* ]] || { echo "LightCompress path must not contain whitespace: $LLMC_REPO" >&2; exit 2; }
[[ "$LLMC_VENV" != *[[:space:]]* ]] || { echo "LightCompress venv path must not contain whitespace: $LLMC_VENV" >&2; exit 2; }
[[ -f "$MATRIX" ]] || { echo "matrix not found: $MATRIX" >&2; exit 2; }

cd -- "$ROOT"
PYTHON="$ROOT/.venv-seq/bin/python"
[[ "$DRY" == 1 || -x "$PYTHON" ]] || { echo "SEQ Python not found: $PYTHON" >&2; exit 2; }
CALIB="$ROOT/calibration_prompts.json"
SWEEPS="$OUT/sweeps"
LLMC_OUT="$OUT/llmc"
REPORTS="$OUT/reports"
STAGE="$OUT/staging"
mkdir -p "$OUT" "$SWEEPS" "$LLMC_OUT" "$REPORTS" "$STAGE"

mapfile -t MATRIX_MODELS < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["models"]))' "$MATRIX")
if [[ -n "$MODELS_OVERRIDE" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_OVERRIDE"
else
  MODELS=("${MATRIX_MODELS[@]}")
fi
FRACS="$("$PYTHON" -c 'import json,sys; print(",".join(str(x) for x in json.load(open(sys.argv[1]))["fractions"]))' "$MATRIX")"
NONZERO_FRACS="$("$PYTHON" -c 'import json,sys; print(",".join(str(x) for x in json.load(open(sys.argv[1]))["fractions"] if float(x)!=0))' "$MATRIX")"
DET_SEED="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["deterministic_seed"])' "$MATRIX")"
mapfile -t RANDOM_SEEDS < <("$PYTHON" -c 'import json,sys; print("\n".join(str(x) for x in json.load(open(sys.argv[1]))["random_seeds"]))' "$MATRIX")
mapfile -t BASE_SPECS < <("$PYTHON" -c 'import json,sys; print("\n".join("{}|{}".format(x["name"],x["base_bits"]) for x in json.load(open(sys.argv[1]))["bases"]))' "$MATRIX")
mapfile -t SET_SELECTORS < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["set_selectors"]))' "$MATRIX")
mapfile -t GATE_SELECTORS < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["gate_selectors"]))' "$MATRIX")
GATE_ANCHOR_SELECTOR="$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(next(x for x in d["gate_selectors"] if x in d["scalar_selectors"] and x!="random"))' "$MATRIX")"
FULL_SCALAR_SELECTORS="$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); gate=set(d["gate_selectors"]); print(",".join(x for x in d["scalar_selectors"] if x not in gate))' "$MATRIX")"
mapfile -t UNIFORM_BITS < <("$PYTHON" -c 'import json,sys; print("\n".join(str(x) for x in json.load(open(sys.argv[1]))["uniform_hqq_bits"]))' "$MATRIX")
VALUE_BUDGETS="$("$PYTHON" -c 'import json,sys; print(",".join(str(x) for x in json.load(open(sys.argv[1]))["value_tier"]["budgets"]))' "$MATRIX")"
ALL_SIGNALS="$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(d["scalar_selectors"]+d["set_selectors"]+["tier_alloc"]))' "$MATRIX")"

failures=0

exec_cmd() {
  printf 'command:'; printf ' %q' "$@"; echo
  [[ "$DRY" == 1 ]] || "$@"
}

json_status_pass() {
  local path="$1"
  [[ -f "$path" ]] && "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' "$path"
}

sweep_complete() {
  local path="$1" selector_csv="$2" fractions_csv="$3" seed="$4"
  [[ -f "$path" ]] || return 1
  "$PYTHON" -c '
import json,math,sys
p=json.load(open(sys.argv[1])); selectors=set(sys.argv[2].split(",")); fracs={float(x) for x in sys.argv[3].split(",") if x}; seed=int(sys.argv[4])
if p.get("seed") != seed: raise SystemExit(1)
rows=p.get("results",[])
for selector in selectors:
    found={float(r["k_frac"]) for r in rows if r.get("signal")==selector and r.get("k_frac") is not None and math.isfinite(float(r.get("ppl",float("nan")))) and isinstance(r.get("storage"),dict) and not r.get("errors")}
    if not fracs.issubset(found): raise SystemExit(1)
' "$path" "$selector_csv" "$fractions_csv" "$seed"
}

checkpoint_path() {
  local model="$1"
  printf '%s/%s/gptq/artifacts/fake_quant_model' "$LLMC_OUT" "${model##*/}"
}

base_args() {
  local model="$1" base="$2"
  BASE_ARGS=(--base_quantizer "$base")
  if [[ "$base" == "gptq_llmc" ]]; then
    BASE_ARGS+=(--gptq_model_path "$(checkpoint_path "$model")")
  fi
}

run_sweep_cell() {
  local model="$1" base="$2" base_bits="$3" selector_csv="$4" fractions="$5" seed="$6" out_dir="$7" mode="$8"
  local result="$out_dir/channel_pareto.json"
  if [[ "$RESUME" == 1 ]] && sweep_complete "$result" "$selector_csv" "$fractions" "$seed"; then
    echo "reuse valid sweep: $result"
    return 0
  fi
  base_args "$model" "$base"
  local cmd=("$PYTHON" -m seq_core.channel_sweep --model "$model" --backend hqq --base_bits "$base_bits"
             --protect_fracs "$fractions" --seed "$seed" --ppl_mode canonical
             --calibration_prompts "$CALIB" --out_dir "$out_dir" "${BASE_ARGS[@]}")
  if [[ "$mode" == "select" ]]; then
    cmd+=(--select "$selector_csv")
  else
    cmd+=(--signals "$selector_csv")
  fi
  exec_cmd "${cmd[@]}"
}

validate_phase() {
  local name="$1"
  case "$name" in
    preflight) json_status_pass "$OUT/run_manifest.json" ;;
    llmc_smoke) json_status_pass "$OUT/llmc_smoke/config_check/report.json" ;;
    gptq_validate)
      local model
      for model in "${MODELS[@]}"; do json_status_pass "$REPORTS/gptq_${model##*/}.json" || return 1; done ;;
    gate|gate_summary) json_status_pass "$REPORTS/gate_summary.json" ;;
    full_matrix|validate) json_status_pass "$REPORTS/matrix_validation.json" ;;
    comparison) [[ -s "$STAGE/results/final_comparison.json" && -s "$STAGE/docs/COMPARISON.md" ]] ;;
    plots) compgen -G "$STAGE/figures/ppl_vs_actual_bits_*.pdf" >/dev/null ;;
    publish) json_status_pass "$ROOT/results/final_publication.json" ;;
    *) return 1 ;;
  esac
}

run_phase() {
  local name="$1"; shift
  local marker="$OUT/.${name}.complete"
  [[ "$PHASE" == all || "$PHASE" == "$name" ]] || return 0
  if [[ "$RESUME" == 1 && -f "$marker" ]] && validate_phase "$name" >/dev/null 2>&1; then
    echo "skip $name (validated marker exists)"
    return 0
  fi
  rm -f "$marker"
  echo "=== phase: $name ==="
  set +e
  ( set -e; "$@" )
  local phase_status=$?
  set -e
  if [[ "$phase_status" == 0 ]]; then
    [[ "$DRY" == 1 ]] || date -u +%FT%TZ > "$marker"
  else
    failures=$((failures + 1))
    [[ "$FAIL_FAST" == 0 ]] || exit 1
  fi
}

phase_preflight() {
  exec_cmd "$PYTHON" scripts/preflight_final_environment.py --root "$ROOT" --matrix "$MATRIX" \
    --llmc-repo "$LLMC_REPO" --llmc-venv "$LLMC_VENV" --output "$OUT/run_manifest.json"
}

phase_tests() {
  exec_cmd "$PYTHON" -m compileall -q seq_core analysis scripts
  exec_cmd "$PYTHON" -m pytest -q
}

phase_llmc_smoke() {
  exec_cmd "$PYTHON" scripts/run_llmc_w4_baselines.py --model "${MODELS[0]}" --methods gptq \
    --llmc-repo "$LLMC_REPO" --llmc-venv "$LLMC_VENV" --out-dir "$OUT/llmc_smoke" --check-config
}

phase_gptq_baselines() {
  local model
  for model in "${MODELS[@]}"; do
    exec_cmd "$PYTHON" scripts/run_llmc_w4_baselines.py --model "$model" --methods gptq \
      --llmc-repo "$LLMC_REPO" --llmc-venv "$LLMC_VENV" --out-dir "$LLMC_OUT/${model##*/}"
  done
}

phase_gptq_validate() {
  local model summary ppl
  for model in "${MODELS[@]}"; do
    summary="$LLMC_OUT/${model##*/}/gptq/summary.json"
    [[ "$DRY" == 1 || -f "$summary" ]] || { echo "missing GPTQ summary: $summary" >&2; return 1; }
    if [[ "$DRY" == 1 ]]; then ppl="LLMC_PPL"; else ppl="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ppl"])' "$summary")"; fi
    exec_cmd "$PYTHON" scripts/validate_gptq_llmc_base.py --model "$model" \
      --gptq-model-path "$(checkpoint_path "$model")" --llmc-reported-ppl "$ppl" --tolerance 0.25 \
      --output "$REPORTS/gptq_${model##*/}.json"
  done
}

phase_gate() {
  local model base bits selector seed name spec mode fractions
  for model in "${MODELS[@]}"; do
    name="${model##*/}"
    for spec in "${BASE_SPECS[@]}"; do
      IFS='|' read -r base bits <<< "$spec"
      run_sweep_cell "$model" "$base" "$bits" "$GATE_ANCHOR_SELECTOR" "$FRACS" "$DET_SEED" \
        "$SWEEPS/$base/$name/$GATE_ANCHOR_SELECTOR/seed-$DET_SEED" signals
      for selector in "${GATE_SELECTORS[@]}"; do
        [[ "$selector" == "$GATE_ANCHOR_SELECTOR" ]] && continue
        if [[ "$selector" == random ]]; then
          for seed in "${RANDOM_SEEDS[@]}"; do
            run_sweep_cell "$model" "$base" "$bits" random "$NONZERO_FRACS" "$seed" \
              "$SWEEPS/$base/$name/random/seed-$seed" signals
          done
          continue
        fi
        mode=signals
        [[ " ${SET_SELECTORS[*]} " == *" $selector "* ]] && mode=select
        run_sweep_cell "$model" "$base" "$bits" "$selector" "$NONZERO_FRACS" "$DET_SEED" \
          "$SWEEPS/$base/$name/$selector/seed-$DET_SEED" "$mode"
      done
    done
  done
}

phase_gate_summary() {
  exec_cmd "$PYTHON" analysis/final_matrix.py gate --matrix "$MATRIX" --roots "$SWEEPS" \
    --json "$REPORTS/gate_summary.json" --markdown "$REPORTS/GATE_SUMMARY.md"
}

phase_remaining_baselines() {
  local model
  for model in "${MODELS[@]}"; do
    exec_cmd "$PYTHON" scripts/run_llmc_w4_baselines.py --model "$model" --methods rtn,awq \
      --llmc-repo "$LLMC_REPO" --llmc-venv "$LLMC_VENV" --out-dir "$LLMC_OUT/${model##*/}"
  done
}

phase_full_matrix() {
  local model base name bits value_model spec
  for model in "${MODELS[@]}"; do
    name="${model##*/}"
    for spec in "${BASE_SPECS[@]}"; do
      IFS='|' read -r base bits <<< "$spec"
      run_sweep_cell "$model" "$base" "$bits" "$FULL_SCALAR_SELECTORS" "$NONZERO_FRACS" "$DET_SEED" \
        "$SWEEPS/$base/$name/scalars/seed-$DET_SEED" signals
    done
    for bits in "${UNIFORM_BITS[@]}"; do
      if printf '%s\n' "${BASE_SPECS[@]}" | grep -Fxq "hqq|$bits"; then
        echo "reuse shared HQQ-$bits zero-budget anchor for $model"
      else
        local uniform_out="$SWEEPS/uniform/$name/b$bits"
        if ! { [[ "$RESUME" == 1 ]] && sweep_complete "$uniform_out/channel_pareto.json" act_max 0 "$DET_SEED"; }; then
          exec_cmd "$PYTHON" -m seq_core.channel_sweep --model "$model" --backend hqq --base_quantizer hqq \
            --base_bits "$bits" --protect_fracs 0 --signals act_max --seed "$DET_SEED" --ppl_mode canonical \
            --calibration_prompts "$CALIB" --out_dir "$uniform_out"
        fi
      fi
    done
  done
  while IFS= read -r value_model; do
    name="${value_model##*/}"
    local value_out="$SWEEPS/value_tier/$name"
    if ! { [[ "$RESUME" == 1 ]] && [[ -s "$value_out/channel_pareto.json" ]]; }; then
      exec_cmd "$PYTHON" -m seq_core.channel_sweep --model "$value_model" --backend hqq --base_quantizer hqq \
        --base_bits 4 --protect_fracs "$VALUE_BUDGETS" --tier_alloc value --seed "$DET_SEED" \
        --ppl_mode canonical --calibration_prompts "$CALIB" --out_dir "$value_out"
    fi
  done < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["value_tier"]["models"]))' "$MATRIX")
}

phase_validate() {
  exec_cmd "$PYTHON" analysis/final_matrix.py validate --matrix "$MATRIX" --roots "$SWEEPS" \
    --json "$REPORTS/matrix_validation.json" --markdown "$REPORTS/MATRIX_VALIDATION.md"
}

phase_comparison() {
  if [[ "$DRY" != 1 ]] && ! json_status_pass "$REPORTS/matrix_validation.json"; then
    echo "matrix validation must PASS before comparison staging" >&2
    return 1
  fi
  mkdir -p "$STAGE/docs" "$STAGE/results"
  exec_cmd "$PYTHON" scripts/build_final_baseline_index.py --root "$LLMC_OUT" \
    --output "$STAGE/results/llmc_baselines.json"
  exec_cmd "$PYTHON" scripts/enrich_llmc_baseline_storage.py --input "$STAGE/results/llmc_baselines.json" \
    --output "$STAGE/results/llmc_enriched.json"
  exec_cmd "$PYTHON" scripts/augment_final_baselines.py --index "$STAGE/results/llmc_enriched.json" \
    --uniform-root "$SWEEPS/uniform" --anchor-root "$SWEEPS/hqq" \
    --output "$STAGE/results/baselines.json"
  exec_cmd "$PYTHON" analysis/build_comparison.py --sweeps "$SWEEPS" \
    --baselines "$STAGE/results/baselines.json" \
    --signals "$ALL_SIGNALS" \
    --require-sweep-points --out "$STAGE/docs/COMPARISON.md" \
    --csv "$STAGE/results/final_comparison.csv" --json "$STAGE/results/final_comparison.json" \
    --random-replicates "$STAGE/results/final_random_replicates.json"
}

phase_plots() {
  exec_cmd "$PYTHON" analysis/plot_final_results.py --input "$STAGE/results/final_comparison.csv" \
    --output-dir "$STAGE/figures" --require-output
}

phase_publish() {
  if [[ "$PUBLISH" != 1 ]]; then
    echo "publication disabled; rerun with --publish after inspecting staged outputs"
    return 0
  fi
  exec_cmd "$PYTHON" scripts/publish_final_results.py --stage-root "$STAGE" --repo-root "$ROOT" \
    --validation "$REPORTS/matrix_validation.json" --gate-summary "$REPORTS/gate_summary.json"
}

run_phase preflight phase_preflight
run_phase tests phase_tests
run_phase llmc_smoke phase_llmc_smoke
run_phase gptq_baselines phase_gptq_baselines
run_phase gptq_validate phase_gptq_validate
run_phase gate phase_gate
run_phase gate_summary phase_gate_summary
run_phase remaining_baselines phase_remaining_baselines
run_phase full_matrix phase_full_matrix
run_phase validate phase_validate
run_phase comparison phase_comparison
run_phase plots phase_plots
run_phase publish phase_publish

echo "pipeline complete; failures=$failures"
exit "$failures"
