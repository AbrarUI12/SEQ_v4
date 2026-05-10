# OmniQuant Validation Notes

Status: integration scaffold created; adapter dry-run passed; upstream parity has not been run yet.

## Pinned Source

- Upstream repo: `https://github.com/OpenGVLab/OmniQuant`
- Pinned commit: `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`
- Local path: `third_party_quant/OmniQuant/`

On this Windows checkout, Git reports a case-only image collision in `imgs/`. This should not affect the Python implementation files.

## Adapter

The adapter is:

```text
third_party_quant/adapters/omniquant_adapter.py
```

It invokes upstream `main.py` through a subprocess and writes provenance to:

```text
<output_dir>/quant/upstream_provenance.json
```

The adapter intentionally does not reimplement OmniQuant internals.

## Environment

Separate environment spec:

```text
third_party_quant/envs/omniquant-upstream.environment.yml
```

Recommended setup:

```bash
conda env create -f third_party_quant/envs/omniquant-upstream.environment.yml
conda activate omniquant-upstream
cd third_party_quant/OmniQuant
pip install -e .
```

For WSL/Linux, keep caches outside `/mnt/d`:

```bash
export HF_HOME=~/hf-cache
export TORCH_HOME=~/torch-cache
export XDG_CACHE_HOME=~/seq-cache
```

## Dry-Run Validation

From the clean SEQ root:

```powershell
.\.venv-seq\Scripts\python.exe third_party_quant\adapters\omniquant_adapter.py `
  --model facebook/opt-125m `
  --output_dir .\artifacts\omniquant_dry_run `
  --dry_run
```

This checks command construction and provenance writing. It does not validate upstream import/runtime parity.

Dry-run completed on the current Windows host with:

- model: `facebook/opt-125m`
- settings: `W4A16`, `group_size=128`, `lwc=True`, `let=True`, `epochs=20`, `nsamples=128`
- provenance: `artifacts/omniquant_dry_run/quant/upstream_provenance.json`
- detected GPU: `NVIDIA GeForce RTX 5090`
- detected upstream commit: `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`
- pin match: `true`

## Phase 1: Installation Validation

Not yet complete. `conda`/`mamba` and Python 3.10 were not available on the current Windows host, so the separate upstream environment was not created here.

Required checks:

- upstream OmniQuant imports in `omniquant-upstream`
- `python third_party_quant/OmniQuant/main.py --help` works in that env
- bug-fixed AutoGPTQ is installed if using `--real_quant`
- adapter dry run succeeds

## Phase 2: Single-Model Upstream Parity

Not yet complete.

Best first target from upstream-supported families:

```text
facebook/opt-125m
```

This is small enough for installation validation and belongs to the upstream-supported OPT family.

Example upstream-parity command shape:

```bash
python main.py \
  --model facebook/opt-125m \
  --epochs 20 \
  --output_dir ./log/opt-125m-w4a16g128 \
  --eval_ppl \
  --wbits 4 \
  --abits 16 \
  --group_size 128 \
  --lwc \
  --let
```

Remaining work:

- run upstream direct command
- run adapter with equivalent arguments
- compare logs, perplexity, saved artifacts, and provenance
- record result here

## Phase 3: SEQ Harness Comparison

The clean SEQ pipeline can now launch the pinned upstream adapter and evaluate the saved OmniQuant model under:

```text
runs/<run_id>/compare_methods/omniquant/
```

Example from the repo root:

```bash
python -m seq_core.pipeline \
  --experiment main \
  --model_name meta-llama/Llama-3.2-1B \
  --compare-methods omniquant
```

Prerequisites still apply:

- `third_party_quant/OmniQuant/` must exist at the pinned commit
- the OmniQuant environment/python must be available
- if `let: true`, you must provide upstream `act_scales` and `act_shifts`

You can point the SEQ pipeline at a separate OmniQuant environment with:

```bash
export OMNIQUANT_PYTHON=/path/to/omniquant-env/bin/python
export OMNIQUANT_UPSTREAM_DIR=/path/to/OmniQuant
export OMNIQUANT_CACHE_DIR=~/seq-cache/omniquant
```
