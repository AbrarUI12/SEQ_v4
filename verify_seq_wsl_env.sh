#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/SEQ_Clean"
source .venv-seq/bin/activate

python - <<'PY'
import importlib.util
import sys

print("python_executable:", sys.executable)
print("python_version:", sys.version)

for name in [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
    "lm_eval",
]:
    try:
        module = __import__(name)
        print(name, "OK", getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(name, "FAIL", repr(exc))

try:
    import torch

    print("torch_cuda_build:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("vram_gb:", round(props.total_memory / 1024**3, 2))
except Exception as exc:
    print("torch_cuda_check_FAIL:", repr(exc))

for name in ["gptqmodel", "auto_gptq"]:
    print(name, "found:", importlib.util.find_spec(name) is not None)
PY

python - <<'PY'
import torch
import bitsandbytes as bnb

print("bitsandbytes_cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    layer = bnb.nn.Linear8bitLt(16, 8).cuda()
    x = torch.randn(2, 16, device="cuda", dtype=torch.float16)
    y = layer(x)
    print("bitsandbytes Linear8bitLt OK:", tuple(y.shape), y.dtype)
PY
