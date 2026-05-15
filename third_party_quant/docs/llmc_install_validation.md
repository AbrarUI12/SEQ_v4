# LLMC Install And Validation

Date: 2026-05-12

## Summary

This pass installed and validated LightCompress (historical name: LLMC) as a separate external baseline tool without modifying SEQ algorithm code.

Scope completed:

- detected platform and toolchain
- cloned LightCompress into a side-by-side repo
- created a separate Python 3.11 virtual environment
- installed LLMC requirements and `lm_eval[hf]`
- validated imports, CUDA visibility, and required LLMC files
- generated two smoke-test configs
- generated an optional WSL smoke-runner script

Scope intentionally not done:

- no SEQ algorithm changes
- no `run_compare_matrix.py` integration
- no `experiments.yaml` changes
- no real quantization run

## Environment

| Item | Value |
| --- | --- |
| SEQ repo root | `E:\SEQ_Clean` |
| Preferred runtime used | WSL2 Ubuntu |
| Native host OS | Windows 11 Pro |
| WSL distro | Ubuntu 24.04.4 LTS |
| WSL kernel | `6.6.87.2-microsoft-standard-WSL2` |
| Native Windows detected | Yes |
| WSL detected | Yes |
| `nvidia-smi` on Windows | Available |
| `nvidia-smi` in WSL | Available |
| GPU visible in WSL | Yes |
| GPU | `NVIDIA GeForce RTX 5090` |
| Conda | Not found |
| Docker | Not found |
| Windows Python | `3.12.3` |
| WSL Python 3.11 | `3.11.15` |
| Python chosen for LLMC | `3.11.15` |

## Repository Setup

| Item | Value |
| --- | --- |
| LightCompress repo path (Windows) | `E:\LightCompress` |
| LightCompress repo path (WSL) | `/mnt/e/LightCompress` |
| LightCompress remote | `https://github.com/ModelTC/LightCompress.git` |
| LightCompress branch | `main` |
| LightCompress commit | `f68af66a4880291271c4803186a8bea12b96a5ef` |
| Side-by-side with SEQ | Yes |
| Cloned inside SEQ repo | No |

## Python Environment

| Item | Value |
| --- | --- |
| Venv path (Windows) | `E:\LightCompress\.venv-llmc` |
| Venv path (WSL) | `/mnt/e/LightCompress/.venv-llmc` |
| Activation command | `source /mnt/e/LightCompress/.venv-llmc/bin/activate` |
| Pip upgraded | Yes |
| `requirements.txt` installed | Yes |
| `lm_eval[hf]` installed | Yes |

Installed package versions verified:

| Package | Version |
| --- | --- |
| `torch` | `2.11.0` |
| `transformers` | `5.8.0` |
| `datasets` | `2.16.1` |
| `accelerate` | `1.13.0` |
| `lm_eval` | `0.4.12` |

## Import Validation

| Check | Result |
| --- | --- |
| `import torch` | Pass |
| `import transformers` | Pass |
| `import datasets` | Pass |
| `import accelerate` | Pass |
| `import lm_eval` | Pass |
| `import llmc` | Pass |
| `torch.cuda.is_available()` | `True` |
| GPU name from `torch` | `NVIDIA GeForce RTX 5090` |
| `llmc/__main__.py` present | Pass |
| `scripts/run_llmc.sh` present | Pass |

## LLMC File Validation

| File | Result |
| --- | --- |
| `E:\LightCompress\llmc\__main__.py` | Pass |
| `E:\LightCompress\scripts\run_llmc.sh` | Pass |
| `E:\LightCompress\configs\quantization\methods\GPTQ\gptq_w_only.yml` | Pass |
| `E:\LightCompress\configs\quantization\methods\SmoothQuant\smoothquant_w_a.yml` | Pass |
| `E:\LightCompress\configs\quantization\methods\Awq\awq_w_only.yml` | Pass |
| `E:\LightCompress\configs\quantization\methods\RTN\rtn_w_only.yml` | Pass |

## Smoke Configs

Generated config files:

| Config | Path |
| --- | --- |
| GPTQ smoke | `E:\SEQ_Clean\third_party_quant\llmc_smoke_configs\gptq_opt125m_smoke.yml` |
| SmoothQuant smoke | `E:\SEQ_Clean\third_party_quant\llmc_smoke_configs\smoothquant_opt125m_smoke.yml` |

Generated save paths embedded in the configs:

| Method | Save path |
| --- | --- |
| GPTQ | `/mnt/e/SEQ_Clean/results/llmc_smoke/gptq_opt125m_tiny_128` |
| SmoothQuant | `/mnt/e/SEQ_Clean/results/llmc_smoke/smoothquant_opt125m_tiny_128` |

Notes:

- save paths are absolute WSL paths
- save paths do not currently exist
- LLMC hard-fails if the save path already exists
- the GPTQ smoke config keeps upstream GPTQ quantization settings
- the SmoothQuant smoke config keeps upstream SmoothQuant quantization settings

Method-specific note:

- the GPTQ upstream method config uses `eval_pos: [fake_quant]`, so the smoke config preserves that instead of forcing transformed evaluation

## Generated Smoke Commands

### GPTQ smoke

```bash
cd /mnt/e/LightCompress
source .venv-llmc/bin/activate
export PYTHONPATH=/mnt/e/LightCompress:$PYTHONPATH
torchrun --standalone --nproc_per_node=1 llmc/__main__.py \
  --config /mnt/e/SEQ_Clean/third_party_quant/llmc_smoke_configs/gptq_opt125m_smoke.yml \
  --task_id gptq_opt125m_tiny_128
```

### SmoothQuant smoke

```bash
cd /mnt/e/LightCompress
source .venv-llmc/bin/activate
export PYTHONPATH=/mnt/e/LightCompress:$PYTHONPATH
torchrun --standalone --nproc_per_node=1 llmc/__main__.py \
  --config /mnt/e/SEQ_Clean/third_party_quant/llmc_smoke_configs/smoothquant_opt125m_smoke.yml \
  --task_id smoothquant_opt125m_tiny_128
```

## Optional Runner Script

Generated helper script:

- `E:\SEQ_Clean\third_party_quant\run_llmc_smoke.sh`

Usage:

```bash
bash /mnt/e/SEQ_Clean/third_party_quant/run_llmc_smoke.sh /mnt/e/LightCompress
```

Behavior:

- activates `.venv-llmc`
- exports `PYTHONPATH`
- runs GPTQ smoke first
- runs SmoothQuant smoke second
- tees logs into `results/llmc_smoke/logs/`
- stops on first failure

## OOM fix for LLMC smoke tests

The original GPTQ smoke run was not failing due to SEQ changes. `dmesg` confirmed that the Linux OOM killer terminated Python during LLMC execution, with Python anonymous RSS reaching roughly 30 GB.

Environment fix applied:

- WSL memory was increased to about `60GiB` RAM
- WSL swap was increased to about `64GiB`

Smoke-test config rules going forward:

- use `wikitext2` for smoke calibration and eval
- do not use `pileval` in smoke configs
- keep smoke calibration at `n_samples=4`
- keep smoke calibration and eval at `seq_len=128`
- set `inference_per_block: true` during eval to reduce memory pressure

Manual smoke command:

```bash
bash third_party_quant/run_llmc_smoke.sh /mnt/e/LightCompress
```

Useful checks:

```bash
free -h
```

```bash
dmesg -T | tail -120
```

## Known Risks

1. LLMC installed successfully, but no real quantization was run yet.
2. `transformers` resolved to `5.8.0`; if an LLMC method later shows compatibility drift, pinning may be needed.
3. Real paper-scale or larger-model runs may still need careful calibration sample counts, sequence lengths, and eval settings even after the WSL memory increase.
4. Real quantization still depends on runtime CUDA behavior, model download success, and method-specific compatibility.
5. LLMC save paths must remain fresh on each real run.

## Next Recommended Step

Do not integrate with `run_compare_matrix.py` yet.

Recommended next action:

1. Run the two smoke commands manually in WSL.
2. Confirm that `facebook/opt-125m` downloads and that LLMC completes at least one small method successfully.
3. If both smoke tests pass, start the thin external integration layer under `third_party_quant/` and only then add `run_compare_matrix.py` dispatch.

## Integration Phase Note

The next integration phase keeps `run_compare_matrix.py` as the public entrypoint and adds only a thin LLMC external-baseline layer under `third_party_quant/`.

Phase 1 integration scope:

- `gptq_llmc`
- `smoothquant_llmc`

Still deferred after phase 1:

- `awq_llmc`
- `rtn_llmc`
- `omniquant_llmc`
- backend export and downstream artifact evaluation
