#!/usr/bin/env bash
set -euo pipefail

SEQ_ROOT="${SEQ_ROOT:-/mnt/e/SEQ_Clean}"
LLMC_REPO="${LLMC_REPO:-/mnt/e/LightCompress}"
LLMC_REMOTE="${LLMC_REMOTE:-https://github.com/ModelTC/LightCompress.git}"
LLMC_REF="${LLMC_REF:-f68af66a4880291271c4803186a8bea12b96a5ef}"
LLMC_VENV="${LLMC_VENV:-$LLMC_REPO/.venv-llmc}"
PY="${PYTHON_BIN:-python3.11}"
CREATE_SEQ_VENV="${CREATE_SEQ_VENV:-0}"
SEQ_PY="${SEQ_PYTHON_BIN:-python3.12}"

pick_py() {
  command -v "$PY" >/dev/null 2>&1 && return
  for c in python3.12 python3.11 python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; return; }; done
  echo "Install python3.11 or python3.12 in WSL." >&2; exit 1
}

[[ -d "$SEQ_ROOT" ]] || { echo "SEQ repo not found: $SEQ_ROOT" >&2; exit 1; }
pick_py

if [[ ! -d "$LLMC_REPO/.git" ]]; then
  mkdir -p "$(dirname "$LLMC_REPO")"
  git clone "$LLMC_REMOTE" "$LLMC_REPO"
elif [[ -n "$LLMC_REF" ]]; then
  (
    cd "$LLMC_REPO"
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "LightCompress has local changes; not changing ref." >&2
    else
      git fetch --all --tags
      git checkout "$LLMC_REF"
    fi
  )
fi

[[ -f "$LLMC_VENV/bin/activate" ]] || "$PY" -m venv "$LLMC_VENV"
# shellcheck disable=SC1091
source "$LLMC_VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$LLMC_REPO/requirements.txt"
python -m pip install -r "$SEQ_ROOT/requirements.lm_eval.txt"
python -m pip install -r "$SEQ_ROOT/requirements.wsl_compare_extras.txt"
export PYTHONPATH="$LLMC_REPO:${PYTHONPATH:-}"

if [[ "$CREATE_SEQ_VENV" == "1" ]]; then
  command -v "$SEQ_PY" >/dev/null 2>&1 || SEQ_PY="$PY"
  [[ -f "$SEQ_ROOT/.venv-seq/bin/activate" ]] || "$SEQ_PY" -m venv "$SEQ_ROOT/.venv-seq"
  # shellcheck disable=SC1091
  source "$SEQ_ROOT/.venv-seq/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  tmp="$(mktemp)"
  grep -v '^torch==' "$SEQ_ROOT/requirements.clean.lock.txt" > "$tmp"
  python -m pip install -r "$tmp"
  rm -f "$tmp"
  # shellcheck disable=SC1091
  source "$LLMC_VENV/bin/activate"
fi

python - <<'PY'
import importlib, importlib.metadata as m, torch
for p in ["torch","transformers","datasets","accelerate","lm_eval","bitsandbytes","matplotlib","numpy","huggingface-hub"]:
    try: print(f"{p}: {m.version(p)}")
    except m.PackageNotFoundError: print(f"{p}: MISSING")
for mod in ["transformers","datasets","accelerate","lm_eval","bitsandbytes","matplotlib","yaml","numpy","llmc"]:
    importlib.import_module(mod)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
python "$SEQ_ROOT/third_party_quant/validate_llmc_smoke_configs.py"

cat <<EOF

WSL benchmark environment is ready.
Next:
  cd "$SEQ_ROOT"
  source "$LLMC_VENV/bin/activate"
  huggingface-cli login
  bash third_party_quant/run_llmc_smoke.sh "$LLMC_REPO" gptq,smoothquant,awq,rtn
  bash scripts/run_qwen3_4b_wsl_compare.sh
EOF
