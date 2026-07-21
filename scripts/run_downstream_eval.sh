#!/usr/bin/env bash
# W2 — downstream lm-eval at the paper operating points (Track A finding F5).
#
# For each (model, operating point) in configs/downstream_operating_points.json:
#   1. resolve a reloadable HF checkpoint dir
#      - 'hf' points  -> the model hub id (fp16) or the saved GPTQ dir (gptq4)
#      - 'export' pts -> materialize once with seq_core.channel_sweep --save_model_path
#   2. run the lm-evaluation-harness 'paper' task set on it (with --log_samples so the
#      CPU aggregator can compute paired-bootstrap CIs), writing results under
#      runs/final/downstream/<Model>/<point>/lm_eval/ plus a seq_meta.json sidecar.
#
# GPU box only (needs .venv-seq + torch + lm_eval). Use --dry-run anywhere to print
# the exact export/eval commands and resolved paths without importing torch.
#
# Example (3B first, as the config priority orders it):
#   bash scripts/run_downstream_eval.sh --resume
#   bash scripts/run_downstream_eval.sh --models meta-llama/Llama-3.2-3B --limit 200   # smoke
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$ROOT"

CONFIG="$ROOT/configs/downstream_operating_points.json"
MODELS_OVERRIDE=""
POINTS_OVERRIDE=""
OUTROOT="$ROOT/runs/final/downstream"
CKPTROOT=""
TASKS=""
LIMIT=""
DEVICE="cuda:0"
BATCH_SIZE="1"
RESUME=0
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --models) MODELS_OVERRIDE="$2"; shift 2 ;;
    --points) POINTS_OVERRIDE="$2"; shift 2 ;;
    --output-root) OUTROOT="$2"; shift 2 ;;
    --ckpt-root) CKPTROOT="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) grep '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }
[[ -n "$CKPTROOT" ]] || CKPTROOT="$OUTROOT/checkpoints"

# .venv-seq python on the GPU box; plain python3 is enough for --dry-run parsing.
PYTHON="$ROOT/.venv-seq/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  if [[ "$DRY" == 1 ]]; then PYTHON="$(command -v python3 || command -v python)"; else
    echo "SEQ python not found: $PYTHON (run on the GPU box, or pass --dry-run)" >&2; exit 2
  fi
fi

CALIB="$ROOT/$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("calibration_prompts","calibration_prompts.json"))' "$CONFIG")"
SEED="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("deterministic_seed",1234))' "$CONFIG")"

# Task set: reuse the benchmarks 'paper' preset unless overridden, so this stays in
# lockstep with benchmarks/eval_config.py.
if [[ -z "$TASKS" ]]; then
  TASKS="$("$PYTHON" - <<'PY' 2>/dev/null || true
try:
    from benchmarks.eval_config import default_lm_eval_presets as d
    print(",".join(d()["paper"]["tasks"]))
except Exception:
    pass
PY
)"
fi
[[ -n "$TASKS" ]] || TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai"

# Models in config priority order (lower 'priority' first); optional --models filter.
mapfile -t ALL_MODELS < <("$PYTHON" -c '
import json,sys
d=json.load(open(sys.argv[1]))["models"]
print("\n".join(sorted(d, key=lambda m: d[m].get("priority", 99))))' "$CONFIG")
if [[ -n "$MODELS_OVERRIDE" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_OVERRIDE"
else
  MODELS=("${ALL_MODELS[@]}")
fi

exec_cmd() { printf 'command:'; printf ' %q' "$@"; echo; [[ "$DRY" == 1 ]] || "$@"; }

# cfg <model> <key-path...> : read a JSON scalar from the config
cfg() { "$PYTHON" - "$CONFIG" "$@" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
cur=d
for k in sys.argv[2:]:
    cur = cur[int(k)] if isinstance(cur, list) else cur[k]
print("" if cur is None else cur)
PY
}

echo "=== downstream eval: tasks=[$TASKS] device=$DEVICE limit=${LIMIT:-full} resume=$RESUME dry=$DRY ==="
failures=0

for model in "${MODELS[@]}"; do
  mbase="${model##*/}"
  gptq_base="$ROOT/$(cfg "models" "$model" "gptq_base")"

  if [[ -n "$POINTS_OVERRIDE" ]]; then
    IFS=',' read -r -a POINTS <<< "$POINTS_OVERRIDE"
  else
    mapfile -t POINTS < <("$PYTHON" -c '
import json,sys
print("\n".join(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["points"]))' "$CONFIG" "$model")
  fi

  for point in "${POINTS[@]}"; do
    echo "--- $mbase :: $point ---"
    kind="$(cfg "point_defs" "$point" "kind")"
    point_out="$OUTROOT/$mbase/$point"
    lm_out="$point_out/lm_eval"
    mkdir -p "$point_out"

    # ---- 1. resolve a reloadable pretrained path --------------------------- #
    case "$kind" in
      hf)
        source="$(cfg "point_defs" "$point" "source")"
        if [[ "$source" == "model" ]]; then pretrained="$model"; else pretrained="$gptq_base"; fi
        if [[ "$source" == "gptq_base" && "$DRY" != 1 && ! -d "$pretrained" ]]; then
          echo "  MISSING GPTQ base dir: $pretrained (run the GPTQ baseline phase first)" >&2
          failures=$((failures+1)); continue
        fi
        ;;
      export)
        pretrained="$CKPTROOT/$mbase/$point"
        if [[ "$RESUME" == 1 && -f "$pretrained/seq_export_manifest.json" ]]; then
          echo "  reuse exported checkpoint: $pretrained"
        else
          bq="$(cfg "point_defs" "$point" "base_quantizer")"
          bb="$(cfg "point_defs" "$point" "base_bits")"
          mode="$(cfg "point_defs" "$point" "mode")"
          signal="$(cfg "point_defs" "$point" "signal")"
          kfrac="$(cfg "point_defs" "$point" "k_frac")"
          export_cmd=("$PYTHON" -m seq_core.channel_sweep --model "$model" --backend hqq
                      --base_bits "$bb" --protect_fracs "$kfrac" --seed "$SEED"
                      --ppl_mode canonical --calibration_prompts "$CALIB"
                      --base_quantizer "$bq"
                      --out_dir "$pretrained/_sweep"
                      --save_model_path "$pretrained" --save_signal "$signal" --save_k_frac "$kfrac")
          if [[ "$bq" == "gptq_llmc" ]]; then
            if [[ "$DRY" != 1 && ! -d "$gptq_base" ]]; then
              echo "  MISSING GPTQ base dir: $gptq_base (needed to build $point)" >&2
              failures=$((failures+1)); continue
            fi
            export_cmd+=(--gptq_model_path "$gptq_base")
          fi
          if [[ "$mode" == "select" ]]; then export_cmd+=(--select "$signal"); else export_cmd+=(--signals "$signal"); fi
          exec_cmd "${export_cmd[@]}"
        fi
        ;;
      *) echo "  unknown point kind: $kind" >&2; failures=$((failures+1)); continue ;;
    esac

    # ---- 2. run the lm-eval-harness paper task set ------------------------- #
    # Guard resume against a limit change: a `--limit N` smoke run writes partial
    # results, and a later full `--resume` (or a different N) must NOT silently
    # reuse them. Compare the limit recorded in the prior run's seq_meta.json
    # against the current one; only reuse when they match.
    cur_limit="${LIMIT:-full}"
    prev_limit="$("$PYTHON" - "$point_out/seq_meta.json" <<'PY'
import json, sys
try:
    v = json.load(open(sys.argv[1])).get("limit")
except Exception:
    v = None
print("full" if v in (None, "") else int(v))
PY
)"
    have_results="$(find "$lm_out" -name '*.json' 2>/dev/null | head -1)"
    if [[ "$RESUME" == 1 && -n "$have_results" && "$prev_limit" == "$cur_limit" ]]; then
      echo "  reuse lm-eval results under: $lm_out (limit=$cur_limit)"
    else
      if [[ "$RESUME" == 1 && -n "$have_results" && "$prev_limit" != "$cur_limit" ]]; then
        echo "  NOT reusing lm-eval results: limit changed (was $prev_limit, now $cur_limit); re-running"
        rm -rf "$lm_out"
      fi
      mkdir -p "$lm_out"
      eval_cmd=("$PYTHON" -m lm_eval --model hf
                --model_args "pretrained=$pretrained,dtype=float16"
                --tasks "$TASKS" --num_fewshot 0 --batch_size "$BATCH_SIZE"
                --device "$DEVICE" --log_samples --output_path "$lm_out")
      [[ -n "$LIMIT" ]] && eval_cmd+=(--limit "$LIMIT")
      exec_cmd "${eval_cmd[@]}"
    fi

    # ---- 3. metadata sidecar for the aggregator ---------------------------- #
    if [[ "$DRY" != 1 ]]; then
      "$PYTHON" - "$CONFIG" "$model" "$point" "$pretrained" "$TASKS" "${LIMIT:-}" > "$point_out/seq_meta.json" <<'PY'
import json,sys
cfg_path, model, point, pretrained, tasks, limit = sys.argv[1:7]
d=json.load(open(cfg_path))
pd=d["point_defs"][point]
exp=d["models"][model].get("expected_ppl",{}).get(point)
json.dump({
  "model": model, "point": point, "kind": pd["kind"],
  "nominal_bits": pd.get("nominal_bits"), "note": pd.get("note"),
  "base_quantizer": pd.get("base_quantizer"), "base_bits": pd.get("base_bits"),
  "signal": pd.get("signal"), "k_frac": pd.get("k_frac"),
  "checkpoint": pretrained, "expected_ppl": exp,
  "tasks": tasks.split(","), "limit": int(limit) if limit else None,
}, open(sys.stdout.fileno(), "w"), indent=2)
PY
    else
      printf '  (dry-run) would write %s\n' "$point_out/seq_meta.json"
    fi
  done
done

echo "downstream eval complete; failures=$failures"
echo "next: python analysis/build_downstream_table.py --root $OUTROOT --config $CONFIG \\"
echo "        --out docs/DOWNSTREAM.md --csv results/downstream.csv --json results/downstream.json"
exit "$failures"
