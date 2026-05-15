#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${AUTO_GPTQ_REPO_URL:-git+https://github.com/ChenMnZ/AutoGPTQ-bugfix.git}"

echo "[1/5] Checking Python environment"
python - <<'PY'
import sys
print(f"python={sys.version.split()[0]}")
try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch is not importable in the active environment. "
        "Create the omniquant-upstream env first."
    ) from exc
print(f"torch={torch.__version__}")
print(f"torch_cuda={getattr(torch.version, 'cuda', None)}")
PY

echo "[2/5] Ensuring AutoGPTQ build helper dependency is present"
python -m pip install gekko

echo "[3/5] Locating CUDA toolkit"
if ! command -v nvcc >/dev/null 2>&1; then
    cat <<'EOF'
error: nvcc was not found on PATH.

OmniQuant --real_quant needs the bug-fixed AutoGPTQ package, and that package
builds a CUDA extension. On WSL/Linux you need a real CUDA toolkit installation,
not only Python CUDA wheels.

Install the CUDA toolkit in WSL so that both of these work:
  command -v nvcc
  ls /usr/local/cuda
EOF
    exit 1
fi

NVCC_PATH="$(command -v nvcc)"
CUDA_HOME_DEFAULT="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-$CUDA_HOME_DEFAULT}"
export PATH="$CUDA_HOME/bin:$PATH"
if [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
if [ -d "$CUDA_HOME/lib" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

echo "nvcc=$NVCC_PATH"
echo "CUDA_HOME=$CUDA_HOME"

echo "[4/5] Installing bug-fixed AutoGPTQ without build isolation"
python -m pip install --no-build-isolation --no-deps "$REPO_URL"

echo "[5/5] Verifying installation"
python - <<'PY'
import os
import torch
import transformers
import auto_gptq
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"auto_gptq={getattr(auto_gptq, '__version__', 'unknown')}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"CUDA_HOME={os.environ.get('CUDA_HOME')}")
PY
