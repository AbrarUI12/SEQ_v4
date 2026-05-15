# LLMC Validation

Date: 2026-05-12

## Scope

This integration keeps `run_compare_matrix.py` as the public entrypoint and keeps LLMC-specific logic under `third_party_quant/`.

Current LLMC method states:

- smoke-passed: `gptq_llmc`, `smoothquant_llmc`, `awq_llmc`, `rtn_llmc`
- implemented but current OPT smoke failed upstream: `llm_int8_llmc`
- recognized but intentionally disabled: `spinquant_llmc`, `omniquant_llmc`

Still out of scope here:

- vLLM export
- instruction benchmarks
- `lm-eval` on LLMC artifacts
- full `meta-llama/Llama-2-7b` runs

## Repo Audit

Audited integration files:

- `run_compare_matrix.py`
- `third_party_quant/adapters/llmc_adapter.py`
- `third_party_quant/llmc_compare.py`
- `third_party_quant/llmc_templates/`
- `third_party_quant/llmc_smoke_configs/`
- `third_party_quant/run_llmc_smoke.sh`
- `third_party_quant/validate_llmc_smoke_configs.py`

Recognized LLMC methods in SEQ after this pass:

- `gptq_llmc`
- `smoothquant_llmc`
- `awq_llmc`
- `rtn_llmc`
- `llm_int8_llmc`
- `spinquant_llmc`
- `omniquant_llmc`

## Upstream Config Audit

Confirmed upstream LLMC config paths:

- RTN:
  `configs/quantization/methods/RTN/rtn_w_only.yml`
- LlmInt8:
  `configs/quantization/methods/LlmInt8/llmint8_w_only.yml`
- QuaRot:
  `configs/quantization/methods/QuaRot/quarot_w_a.yml`
- QuaRot combination docs:
  `docs/en/source/practice/quarot_gptq.md`

What was not found in the stable `main` LightCompress checkout:

- no `configs/quantization/methods/SpinQuant/`
- no `configs/quantization/methods/Spinquant/`
- no `SpinQuant` algorithm class under `llmc/compression/quantization/`

Nearest rotation-based upstream method:

- `Quarot`

Dependency audit:

- `fast_hadamard_transform`: not installed in `/mnt/e/LightCompress/.venv-llmc`

Implication:

- `spinquant_llmc` cannot be mapped safely to a real upstream LLMC method in this checkout
- the nearest rotation path is QuaRot, but QuaRot is a different named algorithm and its docs explicitly require the Hadamard kernel dependency

## SpinQuant Experimental Branch Audit

Stable checkout left untouched:

- repo: `/mnt/e/LightCompress`
- branch: `main`
- commit: `f68af66a4880291271c4803186a8bea12b96a5ef`

Separate experimental checkout used for SpinQuant investigation:

- repo: `/mnt/e/LightCompress-spin`
- branch: `dev_spinquant`
- commit: `15e4a45ca105c84f47e81cb0be6cb93afd60e365`

Confirmed SpinQuant files on the experimental branch:

- `configs/quantization/SpinQuant/spinquant_w4a4.yml`
- `llmc/compression/quantization/spinquant.py`
- `scripts/run_spinquant_llama.sh`

Confirmed method name:

- `SpinQuant`

What the experimental branch shows:

- the published branch is real
- upstream config/script target Llama-family models first
- the algorithm code contains `assert self.config['model']['type'] in ['Opt', 'Llama']`
- the branch requires `fast_hadamard_transform` for its Hadamard rotation path
- the config does not reference external pretrained rotation files
- the training optimizer `SGDG` is implemented locally in the branch

Experimental dependency findings:

- `requirements.txt` on `dev_spinquant` is lightweight and does not include `fast_hadamard_transform`
- `lm_eval[hf]` pulled in `peft 0.19.1`, which was incompatible with the branch's pinned `accelerate 0.31.0`
- removing `peft` fixed that unrelated import problem in the isolated experimental venv
- the isolated experimental venv was later updated from `torch 2.11.0+cu130` to `torch 2.11.0+cu126`
- after that change, `fast_hadamard_transform` installed successfully from `/mnt/e/SEQ_Clean/fast-hadamard-transform` with `pip install --no-build-isolation -v .`
- import check log:
  `results/llmc_spinquant_debug/import_check_after_hadamard_fix.log`

Experimental import snapshot from `/mnt/e/LightCompress-spin/.venv-spin`:

- `torch 2.11.0+cu126`
- `transformers 4.44.2`
- `datasets 2.20.0`
- CUDA available on `NVIDIA GeForce RTX 5090`
- `fast_hadamard_transform`: import OK

Runtime warning after the Hadamard fix:

- PyTorch warns that the current build supports CUDA capabilities through `sm_90`
- the host GPU is `NVIDIA GeForce RTX 5090` with `sm_120`
- PyTorch recommends a newer `12.8` or `13.0` build for this GPU

## Method Notes

### RTN

Upstream behavior:

- method name: `RTN`
- calibration: not required conceptually, but smoke configs still use tiny `wikitext2` calibration settings for consistency with the shared adapter
- fake-quant eval: supported

Template added:

- `third_party_quant/llmc_templates/rtn_w8a16.yml`

Why W8A16:

- LLMC backend docs recommend RTN under W8A16 as the simplest safe baseline
- LLMC docs explicitly say RTN is not the recommended choice for W4A16 accuracy-sensitive use

### LLM.int8

Upstream behavior:

- method name: `LlmInt8`
- config file uses both weight and activation quantization
- threshold field: `quant.special.threshold: 6.0`
- fake-quant eval: supported in config

Template added:

- `third_party_quant/llmc_templates/llm_int8_w8a8.yml`

Current status:

- dry-run works through SEQ
- current real `facebook/opt-125m` smoke fails upstream during fake-quant eval

### SpinQuant

Current status:

- `spinquant_llmc` is recognized by SEQ
- it intentionally returns `not_implemented`

Reason:

- the stable `main` LightCompress checkout used by SEQ still does not contain an upstream SpinQuant config or method
- a separate experimental `dev_spinquant` branch does exist, but it is not yet safe to enable through SEQ
- the experimental branch now imports `fast_hadamard_transform`, but the validated TinyLlama smoke still fails before evaluation on this GPU/runtime combination
- a prior direct `facebook/opt-125m` smoke on the experimental branch also failed in SpinQuant setup before evaluation

Experimental smoke config added:

- `third_party_quant/llmc_smoke_configs/spinquant_opt125m_smoke.yml`
- `third_party_quant/llmc_smoke_configs/spinquant_tinyllama_smoke.yml`

Experimental direct smoke result on `dev_spinquant`:

- command was run directly with `torchrun --standalone --nproc_per_node=1 llmc/__main__.py --config ...` from `/mnt/e/LightCompress-spin`
- no final PPL
- no `llmc_duration_time`
- no finish marker
- log:
  `results/llmc_smoke/logs/spinquant_tinyllama_tiny_128.log`

Exact experimental OPT failure:

- location:
  `llmc/compression/quantization/base_blockwise_quantization.py`
- error:
  `AttributeError: 'OPTConfig' object has no attribute 'intermediate_size'`
- triggering path:
  `self.intermediate_size = self.model.model_config.intermediate_size`

Interpretation:

- the experimental SpinQuant branch is Llama-oriented in practice
- although the code asserts `Opt` and `Llama`, the current OPT path is not actually smoke-safe
- the Hadamard dependency issue is now resolved in the isolated experimental venv
- the current blocker is a runtime kernel mismatch on the RTX 5090 host GPU, not a missing SpinQuant dependency

Exact experimental TinyLlama failure after the Hadamard fix:

- model:
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- stage reached:
  model config load, tokenizer load, dataset download, and calibration sample preparation all completed
- failing path:
  `llmc/models/base_model.py` during `collect_first_block_input`
- error:
  `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`

Interpretation of the TinyLlama failure:

- this is not the earlier OPT `intermediate_size` bug
- this is not a missing `fast_hadamard_transform` import
- the experimental Llama path is real, but the current `torch 2.11.0+cu126` runtime is not safe on this RTX 5090 machine
- until the branch is validated on a compatible PyTorch/CUDA build and direct TinyLlama smoke passes, `spinquant_llmc` should remain disabled

### OmniQuant

Current status:

- `omniquant_llmc` remains recognized but intentionally disabled as `not_implemented`

Known upstream failures:

- OPT step 2:
  `ValueError: not enough values to unpack (expected 3, got 2)`
- TinyLlama step 2:
  `TypeError: cannot unpack non-iterable NoneType object`

## Smoke Configs

Current smoke configs validated in this phase:

- `third_party_quant/llmc_smoke_configs/gptq_opt125m_smoke.yml`
- `third_party_quant/llmc_smoke_configs/smoothquant_opt125m_smoke.yml`
- `third_party_quant/llmc_smoke_configs/awq_opt125m_smoke.yml`
- `third_party_quant/llmc_smoke_configs/rtn_opt125m_smoke.yml`
- `third_party_quant/llmc_smoke_configs/llm_int8_opt125m_smoke.yml`

New smoke configs added:

- `third_party_quant/llmc_smoke_configs/rtn_opt125m_smoke.yml`
- `third_party_quant/llmc_smoke_configs/llm_int8_opt125m_smoke.yml`
- `third_party_quant/llmc_smoke_configs/spinquant_opt125m_smoke.yml`

Current validator rules:

- no `pileval`
- `calib.name == wikitext2`
- `calib.n_samples <= 4`
- `calib.seq_len <= 128`
- `eval.seq_len <= 128`
- `eval.inference_per_block == true`
- `save_vllm == false`
- `save_fake == false`
- `save_trans == false`
- `model.path == facebook/opt-125m`
- `model.tokenizer_mode == slow`

## Dry-Run Results

Command:

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate

python run_compare_matrix.py \
  --models "facebook/opt-125m" \
  --methods "rtn_llmc,llm_int8_llmc,spinquant_llmc" \
  --benchmarks "ppl" \
  --experiments_file experiments.smoke.yaml \
  --output_dir results \
  --llmc_repo /mnt/e/LightCompress \
  --llmc_venv /mnt/e/LightCompress/.venv-llmc \
  --llmc_dry_run \
  --llmc_save_mode none \
  --llmc_calib_samples 4 \
  --llmc_calib_seq_len 128 \
  --llmc_eval_seq_len 128
```

Observed result:

- `rtn_llmc`: `dry_run`
- `llm_int8_llmc`: `dry_run`
- `spinquant_llmc`: `not_implemented`
- no crash

Dry-run summary:

- `results/compare_real__opt-125m__rtn_llmc-llm_int8_llmc-spinquant_llmc__ppl__20260512_173014_699032/global_summary.csv`

## Real Smoke Results

Command:

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate

python run_compare_matrix.py \
  --models "facebook/opt-125m" \
  --methods "rtn_llmc,llm_int8_llmc" \
  --benchmarks "ppl" \
  --experiments_file experiments.smoke.yaml \
  --output_dir results \
  --llmc_repo /mnt/e/LightCompress \
  --llmc_venv /mnt/e/LightCompress/.venv-llmc \
  --llmc_save_mode none \
  --llmc_calib_samples 4 \
  --llmc_calib_seq_len 128 \
  --llmc_eval_seq_len 128
```

Run root:

- `results/compare_real__opt-125m__rtn_llmc-llm_int8_llmc__ppl__20260512_173050_233400/`

### RTN

- status: `success`
- PPL: `66.88150024414062`
- duration: `128.38108563423157 s`
- finish marker found: `--- llmc finished ---`

Files:

- summary:
  `results/compare_real__opt-125m__rtn_llmc-llm_int8_llmc__ppl__20260512_173050_233400/facebook_opt-125m/rtn_llmc/summary.json`
- log:
  `results/compare_real__opt-125m__rtn_llmc-llm_int8_llmc__ppl__20260512_173050_233400/facebook_opt-125m/rtn_llmc/logs/combined.log`

### LLM.int8

- status: `failed`
- no final PPL
- no finish marker

Observed upstream failure on `facebook/opt-125m`:

- location:
  `llmc/compression/quantization/llmint8.py`
- error:
  `IndexError: tuple index out of range`
- failing path:
  `fp_indices = torch.where(tmp >= self.threshold)[1]`

Interpretation:

- the current LLMC `LlmInt8` path assumes an activation shape that does not hold in this OPT fake-quant eval path
- this is an upstream LLMC method failure, not a SEQ core logic issue

Files:

- summary:
  `results/compare_real__opt-125m__rtn_llmc-llm_int8_llmc__ppl__20260512_173050_233400/facebook_opt-125m/llm_int8_llmc/summary.json`
- log:
  `results/compare_real__opt-125m__rtn_llmc-llm_int8_llmc__ppl__20260512_173050_233400/facebook_opt-125m/llm_int8_llmc/logs/combined.log`

## Current Recommendation

Methods safe to recommend right now:

- `base`
- `seq`
- `gptq_llmc`
- `smoothquant_llmc`
- `awq_llmc`
- `rtn_llmc`

Methods not recommended right now:

- `llm_int8_llmc`
  current OPT smoke failed upstream
  - `spinquant_llmc`
    stable checkout still has no upstream SpinQuant method/config, and the experimental `dev_spinquant` branch still fails direct smoke on this host
- `omniquant_llmc`
  remains disabled due upstream failures

## Final Llama-2-7B PPL-First Command

Do not run this in this task.

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate

python run_compare_matrix.py \
  --models "meta-llama/Llama-2-7b" \
  --methods "base,seq,gptq_llmc,smoothquant_llmc,awq_llmc,rtn_llmc" \
  --benchmarks "ppl" \
  --experiments_file experiments.smoke.yaml \
  --output_dir results \
  --llmc_repo /mnt/e/LightCompress \
  --llmc_venv /mnt/e/LightCompress/.venv-llmc \
  --llmc_model_type Llama \
  --llmc_save_mode none \
  --llmc_calib_dataset wikitext2 \
  --llmc_eval_dataset wikitext2 \
  --llmc_calib_samples 32 \
  --llmc_calib_seq_len 512 \
  --llmc_eval_seq_len 512
```

Stronger later command:

```bash
python run_compare_matrix.py \
  --models "meta-llama/Llama-2-7b" \
  --methods "base,seq,gptq_llmc,smoothquant_llmc,awq_llmc,rtn_llmc" \
  --benchmarks "ppl" \
  --experiments_file experiments.yaml \
  --output_dir results \
  --llmc_repo /mnt/e/LightCompress \
  --llmc_venv /mnt/e/LightCompress/.venv-llmc \
  --llmc_model_type Llama \
  --llmc_save_mode none \
  --llmc_calib_dataset wikitext2 \
  --llmc_eval_dataset wikitext2 \
  --llmc_calib_samples 128 \
  --llmc_calib_seq_len 512 \
  --llmc_eval_seq_len 2048
```

## Inspect Results

Useful commands:

```bash
grep -R -E "EVAL: ppl|llmc_duration_time|--- llmc finished ---|IndexError|not_implemented" results | tail -60
```

```bash
find results -name global_summary.csv -printf "%T@ %p\n" | sort -n | tail
```
