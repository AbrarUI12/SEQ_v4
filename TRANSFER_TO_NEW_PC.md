# SEQ Transfer Guide

This file is the handoff guide for moving the current SEQ + LLMC + lm-evaluation-harness workflow to a new machine without losing the working setup, command history, or benchmark interpretation rules.

Key source refs used while writing this guide:

- `run_compare_matrix.py:266-291`, `339-347`, `373-421`, `470-521`, `541-555`, `654-660`, `783-870`, `874-902`, `905-928`, `1004-1250`
- `benchmarks/seq_lm_eval.py:27-51`, `54-60`, `116-149`, `179-213`, `216-224`, `249-285`, `288-389`
- `reports/lm_eval_command_verification.md:5-10`, `24-37`, `84-107`, `143-205`, `215-221`, `232-245`, `247-289`, `291-417`
- `reports/lm_eval_harness_audit.md:20-24`, `31-39`, `46-48`
- `experiments.smoke.yaml:6-8`, `63-91`, `130-139`
- `third_party_quant/llmc_compare.py:325-327`, `337-357`, `455-456`, `485`, `570`, `648`, `663-680`
- `third_party_quant/adapters/llmc_adapter.py:178-199`, `428-452`
- `requirements.clean.lock.txt:1`, `23`, `54`
- `requirements.lm_eval.txt:1`
- `docs/LM_EVAL_INTEGRATION.md:7-11`, `57-81`, `111`
- `SETUP_CLEAN_SEQ.md:81-127`, `132-146`, `250-256`
- `E:/LightCompress/docs/en/source/quickstart.md:4-8`
- `E:/LightCompress/docs/en/source/advanced/model_test_v1.md:156-179`

## 1. What this project is

SEQ is the main research codebase for studying LLM quantization and efficiency tradeoffs. In this repo, the practical goal is not only to quantize models, but to compare quantized and unquantized variants under a consistent evaluation pipeline.

The current benchmarking stack exists to answer three related questions:

- How does the original base model behave?
- How does SEQ behave after its own quantization flow?
- How do external baselines such as direct OmniQuant and LLMC methods compare under the same reporting structure?

That is why the active compare runner tracks `base`, `seq`, direct `omniquant`, and several LLMC-backed methods in one place; see `run_compare_matrix.py:33-44` and `run_compare_matrix.py:1004-1250`.

## 2. Repository layout

Known active paths on this machine:

- SEQ repo, WSL path: `/mnt/e/SEQ_Clean`
- SEQ repo, Windows path: `E:\SEQ_Clean`
- LightCompress / LLMC repo, WSL path: `/mnt/e/LightCompress`
- LightCompress / LLMC repo, Windows path: `E:\LightCompress`
- LLMC venv, WSL path: `/mnt/e/LightCompress/.venv-llmc`
- LLMC venv, Windows-mounted path: `E:\LightCompress\.venv-llmc`

Important SEQ locations:

- Main compare runner: `run_compare_matrix.py`
- lm-eval adapter: `benchmarks/seq_lm_eval.py`
- Smoke config: `experiments.smoke.yaml`
- Results root: `results/`
- Reports root: `reports/`
- LLMC integration layer: `third_party_quant/llmc_compare.py`
- LLMC adapter/config renderer: `third_party_quant/adapters/llmc_adapter.py`
- lm-eval docs note: `docs/LM_EVAL_INTEGRATION.md`
- Optional lm-eval requirement: `requirements.lm_eval.txt`
- Pinned clean SEQ dependency snapshot: `requirements.clean.lock.txt`

Important LightCompress locations:

- Root requirements entrypoint: `E:\LightCompress\requirements.txt`
- Runtime dependency list: `E:\LightCompress\requirements\runtime.txt`
- Quickstart install note: `E:\LightCompress\docs\en\source\quickstart.md:4-8`
- LLMC downstream lm-eval docs: `E:\LightCompress\docs\en\source\advanced\model_test_v1.md:156-179`

Notes:

- `reports/lm_eval_harness_audit.md` exists under `reports/`, not at repo root.
- `docs/LM_EVAL_INTEGRATION.md` exists, but parts of it are stale. In particular, it still says older compare entrypoints are not present; see `docs/LM_EVAL_INTEGRATION.md:111`.

## 3. Environments used

### WSL LLMC environment

Path:

- `/mnt/e/LightCompress/.venv-llmc`

Activation:

```bash
source /mnt/e/LightCompress/.venv-llmc/bin/activate
```

Known status:

- This environment successfully ran lm-evaluation-harness through `run_compare_matrix.py`; see `reports/lm_eval_command_verification.md:347-353`.
- Real harness metrics were produced for `facebook/opt-125m` base `hellaswag`; archived success row:
  `results/compare_real__opt-125m__base__lm-eval-hellaswag__20260519_013520_071842/global_summary.csv`
- This is the recommended environment for current lm-eval runs.

Package version check command:

```bash
python -c "import importlib.metadata as m; print('lm-eval', m.version('lm-eval')); print('transformers', m.version('transformers')); print('huggingface-hub', m.version('huggingface-hub')); print('accelerate', m.version('accelerate'))"
```

Actual versions on this machine:

- `lm-eval 0.4.12`
- `transformers 5.8.0`
- `huggingface-hub 1.14.0`
- `accelerate 1.13.0`

Other validation performed here:

- `python -m lm_eval --help` passed in this env.
- `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"` returned:
  - `2.11.0+cu130`
  - `True`
  - `NVIDIA GeForce RTX 5090`

### Windows system `python.exe`

Known status:

- Not usable for SEQ in the current setup.
- It could not import required packages like `torch` and `lm_eval`.

Observed failures on this machine:

- `python -m lm_eval --help` -> `No module named lm_eval`
- `python run_compare_matrix.py --help` -> `No module named 'torch'`

Do not use bare Windows `python.exe` for SEQ or compare-matrix work on the new PC unless you intentionally recreate a full environment there.

### Windows `.venv-seq`

Known status:

- It had `lm-eval` installed.
- But it had a `transformers` / `huggingface-hub` conflict.
- `transformers 4.57.6` requires `huggingface-hub < 1.0`, but the environment had `huggingface-hub 1.14.0`.
- Therefore the lm-eval Hugging Face backend failed in that environment.

Evidence:

- Version snapshot from this machine:
  - `lm-eval 0.4.11`
  - `transformers 4.57.6`
  - `huggingface-hub 1.14.0`
  - `accelerate 1.12.0`
- Import failure reproduced with:
  `.\.venv-seq\Scripts\python.exe -c "import transformers, huggingface_hub; print(transformers.__version__); print(huggingface_hub.__version__)"`
- The clean lock file expected `huggingface_hub==0.36.2`; see `requirements.clean.lock.txt:23`.

How to check:

```powershell
.\.venv-seq\Scripts\python.exe -c "import importlib.metadata as m; print('lm-eval', m.version('lm-eval')); print('transformers', m.version('transformers')); print('huggingface-hub', m.version('huggingface-hub')); print('accelerate', m.version('accelerate'))"
.\.venv-seq\Scripts\python.exe -m lm_eval --help
.\.venv-seq\Scripts\python.exe -c "import transformers"
```

How to fix:

- Best option: recreate the environment cleanly from the pinned SEQ setup in `SETUP_CLEAN_SEQ.md:81-127` and `requirements.clean.lock.txt:1-59`.
- Acceptable fallback: pin `huggingface-hub` to a compatible `<1.0` version before trusting `transformers 4.57.6`.
- Do not blindly upgrade packages without checking `transformers` compatibility first.

## 4. Environment recreation steps on new PC

### A. WSL / Ubuntu

Recommended target is to make the WSL LLMC venv the main benchmark environment again.

1. Copy or clone SEQ to `/mnt/e/SEQ_Clean` or an equivalent mounted path.
2. Copy or clone LightCompress to `/mnt/e/LightCompress` or an equivalent mounted path.
3. In LightCompress, create a dedicated venv for LLMC work.
   The current working environment is `/mnt/e/LightCompress/.venv-llmc`.
4. Activate it:

```bash
source /mnt/e/LightCompress/.venv-llmc/bin/activate
```

5. Install LightCompress dependencies from repo sources that are actually present:
   - `E:/LightCompress/docs/en/source/quickstart.md:4-8` says:
     `pip install -r requirements.txt`
   - `E:/LightCompress/requirements.txt` points to `requirements/runtime.txt`
   - `E:/LightCompress/requirements/runtime.txt` is the actual runtime list
6. Install SEQ-side dependencies as needed.
   The clean pinned package list is documented in `SETUP_CLEAN_SEQ.md:81-127` and mirrored in `requirements.clean.lock.txt`.
7. Install the optional lm-eval package for SEQ-facing runs:
   - `requirements.lm_eval.txt:1` contains `lm-eval[hf]`
   - `SETUP_CLEAN_SEQ.md:250-256` documents installing it
8. If you need mixed `ppl + lm-eval` from the WSL LLMC venv, verify PPL-side extras too.
   On this machine, mixed routing worked, but `ppl` failed in the LLMC venv because `matplotlib` was missing; see `reports/lm_eval_command_verification.md:353` and the archived mixed run
   `results/compare_real__opt-125m__base__mixed-ppl-hellaswag__20260519_014332_090648/global_summary.csv`.

Useful sanity checks:

```bash
python -m lm_eval --help
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import transformers, yaml, numpy; print('imports ok')"
```

If you also want a dedicated SEQ-only environment, use the pinned install notes in `SETUP_CLEAN_SEQ.md:81-127`.

### B. Hugging Face authentication

Gated models such as `meta-llama/Llama-2-7b-hf` require Hugging Face login and access approval.

Login:

```bash
huggingface-cli login
```

Test access:

```bash
python - <<'PY'
from transformers import AutoConfig
print(AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf").model_type)
PY
```

If this fails on the new PC, fix auth and model-access approval before blaming SEQ, OmniQuant, or LLMC.

### C. GPU/CUDA checks

Run:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Observed on this machine:

- `nvidia-smi` saw an `NVIDIA GeForce RTX 5090`
- WSL LLMC venv saw CUDA successfully with `torch 2.11.0+cu130`

## 5. lm-evaluation-harness integration status

Final state after the fixes in this repo:

- `run_compare_matrix.py` supports real lm-eval for `base`; see `run_compare_matrix.py:339-347` and `1071-1093`.
- SEQ uses `model_quantized/` only if it is reloadable; see `run_compare_matrix.py:654-660`.
- OmniQuant uses `saved_model/` only if it is reloadable; see `run_compare_matrix.py:783-870`.
- LLMC only uses lm-eval if it produces a reloadable HF artifact; see `third_party_quant/llmc_compare.py:325-327` and `run_compare_matrix.py:1197-1213`.
- LLMC with `--llmc_save_mode none` cannot run lm-eval and correctly records
  `lm_eval__reason=llmc_save_mode_none_no_reloadable_artifact`; see `run_compare_matrix.py:1199-1203`.
- Mixed `ppl` plus lm-eval tasks are supported; see `run_compare_matrix.py:266-291`.
- Status-only `ok` rows are rejected; real success requires numeric metrics; see `benchmarks/seq_lm_eval.py:216-224` and `376-384`.
- `--lm_eval_fail_policy raise` aborts non-silently after partial summaries are written; see `benchmarks/seq_lm_eval.py:179-213` and `run_compare_matrix.py:414-421`, `1240-1245`.
- Aliases exist:
  - `--eval_tasks` -> `--benchmarks`
  - `--eval_limit` -> `--lm_eval_limit`
  - `--num_fewshot` -> `--lm_eval_num_fewshot`
  See `run_compare_matrix.py:47-81` and `470-490`.

Real success example:

- `results/compare_real__opt-125m__base__lm-eval-hellaswag__20260519_013520_071842/global_summary.csv`
- That row contains numeric task metrics including:
  - `lm_eval__hellaswag__acc=0.4`
  - `lm_eval__hellaswag__acc_norm=0.6`
- See also `reports/lm_eval_command_verification.md:84-107`, `386-409`.

Important note:

- `docs/LM_EVAL_INTEGRATION.md` contains useful concepts and output descriptions, but some command claims are stale relative to the current runner; see `docs/LM_EVAL_INTEGRATION.md:111` and `reports/lm_eval_harness_audit.md:46-48`.

## 6. Commands that worked

Include these exact command blocks in the transfer package.

### A. Verify package versions

```bash
python -c "import importlib.metadata as m; print('lm-eval', m.version('lm-eval')); print('transformers', m.version('transformers')); print('huggingface-hub', m.version('huggingface-hub')); print('accelerate', m.version('accelerate'))"
```

### B. Verify lm-eval CLI

```bash
python -m lm_eval --help
```

### C. Tiny direct lm-eval smoke

```bash
python -m lm_eval run \
  --model hf \
  --model_args pretrained=facebook/opt-125m,dtype=float16 \
  --tasks hellaswag \
  --device cpu \
  --limit 5 \
  --num_fewshot 0 \
  --batch_size 1 \
  --output_path results/lm_eval_smoke_opt125m
```

### D. Compare-matrix base lm-eval smoke

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate

python run_compare_matrix.py \
  --models "facebook/opt-125m" \
  --device cpu \
  --methods "base" \
  --benchmarks "hellaswag" \
  --lm_eval_limit 5 \
  --lm_eval_num_fewshot 0 \
  --lm_eval_batch_size 1
```

### E. Direct OmniQuant lm-eval command

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate

python run_compare_matrix.py \
  --models "meta-llama/Llama-2-7b-hf" \
  --device auto \
  --methods "omniquant" \
  --eval_tasks "hellaswag,arc_easy,piqa" \
  --lm_eval_batch_size 1 \
  --lm_eval_num_fewshot 0 \
  --degeneracy_mode old
```

### F. Base + SEQ + OmniQuant lm-eval command

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate

TS=$(date +%Y%m%d_%H%M%S)

python run_compare_matrix.py \
  --models "meta-llama/Llama-2-7b-hf" \
  --device auto \
  --methods "base,seq,omniquant" \
  --benchmarks "hellaswag,arc_easy,piqa" \
  --lm_eval_batch_size 1 \
  --lm_eval_num_fewshot 0 \
  --degeneracy_mode old \
  --output_dir "results/${TS}"
```

### G. Base + SEQ + LLMC PPL-only command

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate

TS=$(date +%Y%m%d_%H%M%S)

python run_compare_matrix.py \
  --models "meta-llama/Llama-2-7b-hf" \
  --device auto \
  --methods "base,seq,gptq_llmc,smoothquant_llmc,awq_llmc,rtn_llmc" \
  --benchmarks "ppl" \
  --experiments_file experiments.smoke.yaml \
  --output_dir "results/${TS}" \
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

This command is PPL-only.

- It does not run lm-evaluation-harness because `ppl` alone yields `lm_eval_tasks=[]`; see `run_compare_matrix.py:266-291`, `1071-1214`.
- `32` calibration samples and `512` sequence length are smoke-test settings, not ideal final paper settings.

### H. More paper-worthy LLMC PPL command

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate

TS=$(date +%Y%m%d_%H%M%S)

python run_compare_matrix.py \
  --models "meta-llama/Llama-2-7b-hf" \
  --device auto \
  --methods "base,seq,gptq_llmc,smoothquant_llmc,awq_llmc,rtn_llmc" \
  --benchmarks "ppl" \
  --experiments_file experiments.smoke.yaml \
  --output_dir "results/${TS}" \
  --llmc_repo /mnt/e/LightCompress \
  --llmc_venv /mnt/e/LightCompress/.venv-llmc \
  --llmc_model_type Llama \
  --llmc_save_mode none \
  --llmc_calib_dataset wikitext2 \
  --llmc_eval_dataset wikitext2 \
  --llmc_calib_samples 128 \
  --llmc_calib_seq_len 2048 \
  --llmc_eval_seq_len 2048
```

Before final paper numbers:

- Verify LLMC seed control and logging.
- The current runner does fix a seed, but it is hardcoded to `42` and not user-exposed; see Section 8.

Compatibility note for the archived successful WSL smoke runs:

- The verified `facebook/opt-125m` WSL smokes in `reports/lm_eval_command_verification.md:351-353` used `dtype=float32`.
- If the exact smoke commands above behave differently on the new PC, retry the `facebook/opt-125m` smoke with `--dtype float32` to match the archived successful run.

## 7. How to interpret results

Output structure:

```text
results/
  <run>/
    global_summary.csv
    global_summary.json
    metadata.json
    <model_slug>/
      <method>/
        summary.json
        lm_eval/
          lm_eval_command.txt
          lm_eval_raw.json
          lm_eval_summary.json
          lm_eval_status.json
          lm_eval_stdout.txt
          lm_eval_stderr.txt
```

Evidence:

- `benchmarks/seq_lm_eval.py:27-51`
- `run_compare_matrix.py:1028-1061`
- `run_compare_matrix.py:1248-1249`

lm-eval success criteria:

- `lm_eval__status=ok`
- `lm_eval__tasks` contains the requested task list
- At least one numeric metric column exists

Examples of valid numeric lm-eval columns:

- `lm_eval__hellaswag__acc`
- `lm_eval__hellaswag__acc_norm`
- `lm_eval__arc_easy__acc`
- `lm_eval__piqa__acc`

Do not count as lm-eval success:

- `lm_eval__status=skipped`
- `lm_eval__status=error`
- `lm_eval__status=ok` but no numeric task metrics

Why this rule exists:

- `benchmarks/seq_lm_eval.py:216-224` checks for numeric metrics
- `benchmarks/seq_lm_eval.py:376-384` turns missing metrics into `lm_eval_no_numeric_metrics`

PPL success criteria:

- PPL columns should exist in `global_summary.csv`
- LLMC PPL comes from LLMC logs / adapter path, not from lm-eval

Evidence:

- `run_compare_matrix.py:931-974`
- `third_party_quant/llmc_compare.py:303-334`

Concrete examples:

- Real lm-eval success:
  `results/compare_real__opt-125m__base__lm-eval-hellaswag__20260519_013520_071842/global_summary.csv`
- Mixed route where lm-eval succeeded but PPL failed because `matplotlib` was missing:
  `results/compare_real__opt-125m__base__mixed-ppl-hellaswag__20260519_014332_090648/global_summary.csv`

## 8. LLMC PPL details

Current LLMC PPL smoke setup:

```text
--llmc_calib_dataset wikitext2
--llmc_eval_dataset wikitext2
--llmc_calib_samples 32
--llmc_calib_seq_len 512
--llmc_eval_seq_len 512
```

Meaning:

- calibration dataset: WikiText-2
- evaluation dataset: WikiText-2
- calibration samples: 32
- calibration sequence length: 512 tokens
- evaluation sequence length: 512 tokens
- `save_mode none`: no reloadable model for lm-eval

Why this is useful:

- Good for smoke tests and bringing the pipeline up quickly.
- Not good enough for final paper numbers by default.

For final paper numbers:

- Prefer `128` or `256` calibration samples if hardware allows.
- Prefer sequence length `2048` if hardware allows.
- Keep smoke settings clearly labeled as smoke settings.

LLMC seed control status:

- `run_compare_matrix.py` does not expose a `--llmc_seed` flag; see `run_compare_matrix.py:503-515`.
- `_run_llmc_method()` currently hardcodes `seed=42`; see `run_compare_matrix.py:884-902`, especially `897`.
- `third_party_quant/llmc_compare.py` receives that seed and forwards it into `calib_seed`; see `third_party_quant/llmc_compare.py:351`, `485`, `570`, `648`.
- `third_party_quant/adapters/llmc_adapter.py:178-185` writes the seed into rendered config as `config["calib"]["seed"]`.
- Existing rendered configs do show seed `42`; example:
  `results/compare_real__Llama-2-7b-hf__base-seq-gptq_llmc-smoothquant_llmc-awq_llmc-rtn_llmc__ppl__20260512_212547_237127/meta-llama_Llama-2-7b-hf/gptq_llmc/rendered_config.yml`
- Existing LLMC logs also show seed `42`; example:
  `results/llmc_smoke/logs/gptq_opt125m_smoke_rerun.log`

Practical conclusion:

- LLMC seed is fixed today, not absent.
- LLMC seed is not configurable from the compare-matrix CLI today.
- Before final paper sweeps, add or verify a user-facing `--llmc_seed` flag so seed choice is explicit in commands and metadata.

## 9. What not to do

- Do not use bare Windows `python.exe` for SEQ.
- Do not trust Windows `.venv-seq` for lm-eval until the `huggingface-hub` conflict is fixed.
- Do not include LLMC methods in an lm-eval command while using `--llmc_save_mode none`.
- Do not treat skipped/error lm-eval rows as benchmark success.
- Do not treat `lm_eval__status=ok` without numeric task metrics as success.
- Do not compare internal PPL and lm-eval downstream-task accuracy as if they were the same metric.
- Do not use `32` samples / `512` sequence length as final paper numbers unless you clearly label them as smoke settings.
- Do not rely on `docs/LM_EVAL_INTEGRATION.md` alone for current compare-runner behavior; parts of it are stale.

## 10. Final paper benchmark recommendation

Recommended separation for final tables:

Table 1:

- WikiText-2 PPL
- `base`, `SEQ`, `GPTQ`, `SmoothQuant`, `AWQ`, `RTN`, `OmniQuant` where available

Table 2:

- lm-evaluation-harness downstream tasks
- `base`, `SEQ`, `OmniQuant`
- LLMC only if a reloadable artifact is verified end-to-end

Table 3:

- latency
- memory
- compression ratio

Recommended lm-eval tasks:

- `hellaswag`
- `arc_easy`
- `arc_challenge`
- `piqa`
- `winogrande`
- `mmlu` if compute allows

Why split the tables:

- PPL and downstream accuracy are different kinds of evidence.
- LLMC PPL is currently available even when LLMC lm-eval is intentionally skipped.
- Reloadability of saved quantized artifacts still needs method-by-method validation for paper-safe downstream comparisons.

## 11. Known unresolved items

- Verify LLMC seed control for final sweeps.
  Current state: seed is fixed to `42`, forwarded into LLMC config/logs, but not exposed as a CLI flag.
- Verify whether LLMC `fake` and `trans` save modes produce truly reloadable HF artifacts method by method.
- Verify SEQ saved-model reload semantics for final paper numbers.
- Fix or recreate Windows `.venv-seq` if you want a usable Windows-native SEQ environment.
- Install missing PPL-side dependencies such as `matplotlib` in the WSL LLMC venv if mixed `ppl + lm-eval` is needed there.
- Treat `docs/LM_EVAL_INTEGRATION.md` as partially stale until refreshed.

## 12. Quick checklist for new PC

- [ ] SEQ repo copied
- [ ] LightCompress repo copied
- [ ] WSL works
- [ ] GPU/CUDA visible
- [ ] Hugging Face login done
- [ ] LLaMA access verified
- [ ] LLMC venv activated
- [ ] package versions recorded
- [ ] lm-eval CLI works
- [ ] `facebook/opt-125m` smoke passes
- [ ] base lm-eval compare-matrix smoke passes
- [ ] PPL smoke passes
- [ ] `results/global_summary.csv` contains expected columns

## Suggested first command on the new PC

Start with the lowest-cost proof that the recommended environment is healthy:

```bash
cd /mnt/e/SEQ_Clean
source /mnt/e/LightCompress/.venv-llmc/bin/activate
python -c "import importlib.metadata as m; print('lm-eval', m.version('lm-eval')); print('transformers', m.version('transformers')); print('huggingface-hub', m.version('huggingface-hub')); print('accelerate', m.version('accelerate'))"
```

Then run the compare-matrix base smoke from Section 6D.
