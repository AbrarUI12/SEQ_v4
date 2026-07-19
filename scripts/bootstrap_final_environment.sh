#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LLMC_REPO="/mnt/e/LightCompress"
PINNED_LLMC_COMMIT="86f564ddb1d6548b228c67a10509a4ed7264345c"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llmc-repo) LLMC_REPO="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "$LLMC_REPO" != *[[:space:]]* ]] || { echo "LightCompress path must not contain whitespace: $LLMC_REPO" >&2; exit 2; }
[[ -x "$ROOT/.venv-seq/bin/python" ]] || { echo "missing SEQ venv: $ROOT/.venv-seq" >&2; exit 2; }
if command -v uv >/dev/null; then
  UV=(uv)
elif [[ -x "$ROOT/.venv-seq/bin/uv" ]]; then
  UV=("$ROOT/.venv-seq/bin/uv")
else
  echo "uv is required (install it globally or into $ROOT/.venv-seq)" >&2
  exit 2
fi

"${UV[@]}" pip install --python "$ROOT/.venv-seq/bin/python" 'hqq==0.2.8.post1' pytest

if [[ ! -d "$LLMC_REPO/.git" ]]; then
  [[ ! -e "$LLMC_REPO" ]] || { echo "refusing to replace non-git path: $LLMC_REPO" >&2; exit 2; }
  git clone https://github.com/ModelTC/LightCompress.git "$LLMC_REPO"
fi
git -C "$LLMC_REPO" fetch origin "$PINNED_LLMC_COMMIT"
git -C "$LLMC_REPO" checkout --detach "$PINNED_LLMC_COMMIT"
"${UV[@]}" python install 3.11
"${UV[@]}" venv --python 3.11 "$LLMC_REPO/.venv-llmc"
"${UV[@]}" pip install --python "$LLMC_REPO/.venv-llmc/bin/python" \
  -r "$LLMC_REPO/requirements.txt" \
  -c "$ROOT/configs/llmc_constraints.txt"

echo "SEQ and pinned LightCompress environments are ready"
echo "LLMC repo: $LLMC_REPO"
echo "LLMC venv: $LLMC_REPO/.venv-llmc"
