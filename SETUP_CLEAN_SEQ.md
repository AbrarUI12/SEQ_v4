# Clean SEQ Setup

This file is for setting up a clean folder that runs the main SEQ pipeline:

- `seq_core/pipeline.py`
- `seq_core/entropy_metrics.py`
- `seq_core/precision_policy.py`
- `seq_core/quantize_model.py`
- required benchmark/reporting support used by `seq_core.pipeline`

It does not cover the full comparison-matrix stack or research PTQ baselines.

## 1. Files To Copy

Copy these implementation files:

```text
seq_core/
  __init__.py
  pipeline.py
  entropy_metrics.py
  precision_policy.py
  quantize_model.py
```

Copy these benchmark/reporting support files:

```text
benchmarks/
  __init__.py
  core.py
  evaluation_suite.py
  eval_config.py
  metrics_utils.py
  multiple_choice_eval.py
  plotting.py
  reporting.py
  ppl_eval.py
  seq_lm_eval.py
```

Copy these config/input files:

```text
experiments.yaml
experiments.smoke.yaml
calibration_prompts.json
eval_prompts.json
```

Optional documentation:

```text
README.md
SEQ_METHODOLOGY.md
```

## 2. System Requirements

Recommended:

```text
Python 3.12
NVIDIA GPU with CUDA support
Recent NVIDIA driver compatible with CUDA 12.8 wheels
Enough VRAM for the selected model
```

CPU-only Python can import most files, but the actual SEQ INT4/INT8 quantization path requires `bitsandbytes`, which is expected to run with CUDA.

## 3. Create Environment

### Windows PowerShell

```powershell
cd E:\path\to\clean-seq-folder
py -3.12 -m venv .venv-seq
.\.venv-seq\Scripts\python.exe -m pip install --upgrade pip
```

Install PyTorch CUDA wheel:

```powershell
.\.venv-seq\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Install SEQ dependencies:

```powershell
.\.venv-seq\Scripts\python.exe -m pip install `
  accelerate==1.12.0 `
  bitsandbytes==0.49.1 `
  datasets==4.5.0 `
  hf-xet==1.2.0 `
  matplotlib==3.10.8 `
  numpy==2.2.6 `
  pyyaml==6.0.3 `
  safetensors==0.7.0 `
  sentencepiece==0.2.1 `
  tokenizers==0.22.2 `
  tiktoken==0.12.0 `
  tqdm==4.67.1 `
  transformers==4.57.6
```

### Linux / WSL

```bash
cd /path/to/clean-seq-folder
python3.12 -m venv .venv-seq
source .venv-seq/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  accelerate==1.12.0 \
  bitsandbytes==0.49.1 \
  datasets==4.5.0 \
  hf-xet==1.2.0 \
  matplotlib==3.10.8 \
  numpy==2.2.6 \
  pyyaml==6.0.3 \
  safetensors==0.7.0 \
  sentencepiece==0.2.1 \
  tokenizers==0.22.2 \
  tiktoken==0.12.0 \
  tqdm==4.67.1 \
  transformers==4.57.6
```

## 4. Sanity Checks

Run these before launching SEQ:

```powershell
.\.venv-seq\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
.\.venv-seq\Scripts\python.exe -c "import transformers, bitsandbytes, yaml, numpy; print('imports ok')"
```

For Linux / WSL, use:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import transformers, bitsandbytes, yaml, numpy; print('imports ok')"
```

If `torch.cuda.is_available()` prints `False`, fix CUDA/driver/PyTorch before running the quantization pipeline.

## 5. Run A Smoke Test

Use the small smoke config first:

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml
```

Linux / WSL:

```bash
python -m seq_core.pipeline --experiment main --experiments_file experiments.smoke.yaml
```

Expected output folders:

```text
runs/
reports/
```

Inside the latest run, check for:

```text
config.json
env.json
entropy/weight_entropy.json
entropy/activation_entropy.json
policy/precision_map.json
quant/effective_bits.json
eval_baseline/eval_summary.json
eval_quant/eval_summary.json
```

## 6. Run Main SEQ

```powershell
.\.venv-seq\Scripts\python.exe -m seq_core.pipeline --experiment main
```

Linux / WSL:

```bash
python -m seq_core.pipeline --experiment main
```

## 7. Standalone PPL

This runs only the perplexity calculation path, without compare-matrix methods:

```powershell
.\.venv-seq\Scripts\python.exe -m benchmarks.ppl_eval `
  --models "meta-llama/Llama-3.2-1B" `
  --device auto `
  --ppl_mode canonical `
  --ppl_dataset wikitext2 `
  --ppl_split test `
  --ppl_full_corpus true `
  --ppl_seq_len 2048
```

Linux / WSL:

```bash
python -m benchmarks.ppl_eval \
  --models "meta-llama/Llama-3.2-1B" \
  --device auto \
  --ppl_mode canonical \
  --ppl_dataset wikitext2 \
  --ppl_split test \
  --ppl_full_corpus true \
  --ppl_seq_len 2048
```

Results are written under `ppl_results/` by default.

## 8. Optional Environment Variables

For CUDA memory fragmentation:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

Linux / WSL:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For gated Hugging Face models:

```powershell
$env:HF_TOKEN="your_token_here"
```

Linux / WSL:

```bash
export HF_TOKEN="your_token_here"
```

## 9. Optional Packages Not Needed For Main SEQ

These are useful for comparison/baseline scripts, but not required for the core `seq_core.pipeline` SEQ run:

```text
gptqmodel
optimum
pandas
torchao
triton
```

## 10. Common Failures

`bitsandbytes is required for INT4/INT8 quantization`

Install `bitsandbytes==0.49.1` and make sure CUDA is available through PyTorch.

`datasets_unavailable`

Install `datasets==4.5.0`, or disable PPL/MMLU/zero-shot evaluation in `experiments.yaml`.

Out of memory

Use `experiments.smoke.yaml`, reduce `calibration.seq_len`, reduce `evaluation.max_new_tokens`, reduce PPL examples, or use a smaller model.
