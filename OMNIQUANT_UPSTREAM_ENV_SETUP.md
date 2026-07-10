# OmniQuant Upstream Environment Setup

This guide recreates the `omniquant-upstream` environment on another PC using:

```text
third_party_quant/envs/omniquant-upstream.environment.yml
```

The environment uses Python 3.10, PyTorch, and Transformers 4.x. If you need
OmniQuant `--real_quant`, install the upstream bug-fixed AutoGPTQ fork as a
separate second step after the base environment is created.

## 1. Copy The Files

Copy this repo, or at minimum copy this file:

```text
third_party_quant/envs/omniquant-upstream.environment.yml
```

Then open a terminal in the repo root:

```powershell
cd E:\path\to\SEQ_Clean
```

## 2. System Requirements

Recommended:

```text
Windows 10/11 or Linux
NVIDIA GPU
Recent NVIDIA driver compatible with your PyTorch CUDA wheel
Conda, Mamba, or Micromamba
Git and Git LFS
```

For real AutoGPTQ CUDA-extension builds on Windows, also install:

```text
Microsoft C++ Build Tools, including MSVC v14 or newer
```

For real AutoGPTQ CUDA-extension builds on WSL/Linux, also install a real CUDA
toolkit so these work:

```text
command -v nvcc
ls /usr/local/cuda
```

If the C++ build tools or CUDA toolkit are missing, AutoGPTQ's compiled CUDA
extension may not build.

## 3. Option A: Use Conda Or Mamba

If Conda or Mamba is already installed:

```powershell
conda env create -f third_party_quant/envs/omniquant-upstream.environment.yml
```

or:

```powershell
mamba env create -f third_party_quant/envs/omniquant-upstream.environment.yml
```

Activate it:

```powershell
conda activate omniquant-upstream
```

If the environment already exists, update it:

```powershell
conda env update -n omniquant-upstream -f third_party_quant/envs/omniquant-upstream.environment.yml
```

## 4. Option B: Use Portable Micromamba

Use this when the PC does not have Conda on `PATH`.

Download and extract Micromamba in PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path .tools | Out-Null
Invoke-WebRequest -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile ".tools\micromamba.tar.bz2"
tar -xjf ".tools\micromamba.tar.bz2" -C ".tools"
```

Create the env locally under `.micromamba`:

```powershell
.\.tools\Library\bin\micromamba.exe env create `
  --root-prefix .micromamba `
  -f third_party_quant/envs/omniquant-upstream.environment.yml `
  -y
```

Run commands inside it:

```powershell
.\.tools\Library\bin\micromamba.exe run `
  --root-prefix .micromamba `
  -n omniquant-upstream `
  python --version
```

## 5. Install AutoGPTQ For `--real_quant`

The base environment no longer installs AutoGPTQ inline. That is intentional:
it avoids `pip` build-isolation failures during env creation and makes WSL
prerequisites explicit.

### WSL/Linux

From the repo root:

```bash
bash third_party_quant/scripts/install_autogptq_real_quant.sh
```

That script:

- installs `gekko`
- checks for `nvcc`
- exports `CUDA_HOME`
- installs the bug-fixed AutoGPTQ fork with `--no-build-isolation`
- verifies `torch`, `transformers`, and `auto_gptq`

### Windows

After activating `omniquant-upstream` and installing MSVC Build Tools:

```powershell
python -m pip install gekko
python -m pip install --no-build-isolation --no-deps git+https://github.com/ChenMnZ/AutoGPTQ-bugfix.git
```

## 6. Verify The Install

Run this check:

```powershell
python -c "import torch, torchvision, transformers, auto_gptq; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('transformers', transformers.__version__); print('auto_gptq', getattr(auto_gptq, '__version__', 'unknown'))"
```

For portable Micromamba, wrap it like this:

```powershell
.\.tools\Library\bin\micromamba.exe run `
  --root-prefix .micromamba `
  -n omniquant-upstream `
  python -c "import torch, torchvision, transformers, auto_gptq; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('transformers', transformers.__version__); print('auto_gptq', getattr(auto_gptq, '__version__', 'unknown'))"
```

Expected shape:

```text
Python 3.10.x
torch >= 2.0.0
transformers 4.x
auto_gptq 0.5.0.dev0 or similar
```

`torch.cuda.is_available()` should print `True` on a working NVIDIA CUDA setup.

## 7. AutoGPTQ Troubleshooting

The YAML pins Transformers to `<5` because AutoGPTQ 0.5.x imports APIs that were removed in Transformers 5.x.

If AutoGPTQ installation fails while installing:

```text
git+https://github.com/ChenMnZ/AutoGPTQ-bugfix.git
```

first check whether the rest of the environment was created:

```powershell
conda activate omniquant-upstream
python -c "import torch; print(torch.__version__)"
```

Then reinstall AutoGPTQ without build isolation:

```powershell
python -m pip install --no-build-isolation --no-deps git+https://github.com/ChenMnZ/AutoGPTQ-bugfix.git
```

If that fails on Windows with:

```text
Microsoft Visual C++ 14.0 or greater is required
```

install Microsoft C++ Build Tools and run the command again from a fresh terminal.

If that fails on WSL/Linux with errors mentioning `CUDA_HOME` or missing `nvcc`,
install a real CUDA toolkit in WSL first. Python CUDA wheels alone are not
enough for the AutoGPTQ extension build.

## 8. Quick Package Health Check

After setup:

```powershell
python -m pip check
git --version
git-lfs --version
```

On the local PC used to write this guide, the working versions were:

```text
Python 3.10.20
torch 2.11.0+cu128
torchvision 0.26.0+cu128
transformers 4.57.3
auto_gptq 0.5.0.dev0
```
