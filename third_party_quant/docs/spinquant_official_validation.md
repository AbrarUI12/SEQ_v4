# Official SpinQuant Validation

Date: 2026-05-12

## Scope

This document covers a separate upstream investigation for the official SpinQuant repository:

- upstream repo: `/mnt/e/SpinQuant-official`
- upstream origin: `https://github.com/facebookresearch/SpinQuant.git`
- SEQ repo under evaluation: `/mnt/e/SEQ_Clean`

This work did not modify:

- `/mnt/e/LightCompress`
- SEQ core algorithm logic
- `entropy_metrics.py`
- `precision_policy.py`
- `quantize_model.py`

This work also did not re-enable `spinquant_llmc`.

## Official Repo Snapshot

- repo path: `/mnt/e/SpinQuant-official`
- branch: `main`
- commit: `8f47aa3f00e8662caf1a484153920a07e5281c3a`

Recorded inventory:

- `results/spinquant_official_debug/spinquant_repo_inventory.txt`

## Environment

Requested target:

- isolated SpinQuant venv
- not the LLMC venv
- Python 3.11 first

Observed host state:

- native Windows Python 3.11 was not installed
- WSL Python 3.11 was available as `python3.11`
- native Windows Python was `3.12.3`

Created environment:

- venv path: `/mnt/e/SpinQuant-official/.venv-spinquant`
- interpreter used: WSL `python3.11`

## Dependency Status

Installed successfully in the isolated venv:

- `transformers==4.44.2`
- `accelerate==0.34.2`
- `datasets==2.20.0`
- `sentencepiece`
- `tensorboardX`
- `torch 2.11.0+cu130`

Import snapshot:

- log: `results/spinquant_official_debug/import_check.log`
- CUDA visible: `True`
- device: `NVIDIA GeForce RTX 5090`

## fast-hadamard-transform Status

Official README instruction used:

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git fast-hadamard-transform-spinquant
cd /mnt/e/fast-hadamard-transform-spinquant
source /mnt/e/SpinQuant-official/.venv-spinquant/bin/activate
pip install -v .
```

Result:

- install failed
- failure log: `results/spinquant_official_debug/fast_hadamard_install.log`

Observed failure:

- `pip install -v .` failed while getting wheel build requirements
- the isolated build step raised `ModuleNotFoundError: No module named 'torch'`
- this happened on the documented upstream install path, without local patching

Why this blocks SpinQuant:

- `SpinQuant-official/utils/utils.py` imports `from fast_hadamard_transform import hadamard_transform` at module import time
- direct import check confirms `ModuleNotFoundError("No module named 'fast_hadamard_transform'")`
- `optimize_rotation.py` and `ptq.py` both depend on modules that reach this import path

Conclusion:

- `fast_hadamard_transform` did not install cleanly in the isolated SpinQuant environment
- official SpinQuant is blocked before a real smoke run

## Upstream Script and Argument Audit

Files inspected:

- `/mnt/e/SpinQuant-official/README.md`
- `/mnt/e/SpinQuant-official/scripts/10_optimize_rotation.sh`
- `/mnt/e/SpinQuant-official/scripts/2_eval_ptq.sh`
- `/mnt/e/SpinQuant-official/optimize_rotation.py`
- `/mnt/e/SpinQuant-official/ptq.py`
- `/mnt/e/SpinQuant-official/utils/process_args.py`

Confirmed argument support from upstream code and README:

- `--input_model`
- `--output_rotation_path`
- `--optimized_rotation_path`
- `--output_dir`
- `--logging_dir`
- `--w_bits`
- `--a_bits`
- `--k_bits`
- `--v_bits`
- `--w_groupsize`
- `--rotate`
- `--w_rtn`
- `--per_device_train_batch_size`
- `--per_device_eval_batch_size`
- `--access_token`

Important upstream behavior:

- `optimize_rotation.py` loads `Salesforce/wikitext` training data and writes `R.bin` under `output_rotation_path`
- `ptq.py` evaluates WikiText-2 PPL directly and logs `wiki2 ppl is: ...`
- official scripts set `k_groupsize` and `v_groupsize` explicitly
- the two-step workflow is real and matches the README:
  1. optimize rotation
  2. PTQ eval with `optimized_rotation_path`

## Smoke Model Choice

Requested priority:

1. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
2. `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
3. `meta-llama/Llama-2-7b` only after TinyLlama passes

Compatibility check result:

- both TinyLlama configs load successfully through `AutoConfig`
- `model_type`: `llama`
- `architectures`: `['LlamaForCausalLM']`
- `intermediate_size`: present
- `tie_word_embeddings`: `False`

Interpretation:

- TinyLlama appears structurally compatible with the official Llama-only SpinQuant code path
- the blocker was dependency installation, not an obvious model-family mismatch

## Direct Smoke Result

Direct TinyLlama official SpinQuant smoke status:

- not run

Reason:

- the required `fast_hadamard_transform` dependency did not install cleanly
- official SpinQuant utilities fail to import before step 1 can safely begin

Because of that, the following were intentionally not run:

- `optimize_rotation.py`
- `ptq.py`
- zero-shot benchmarks
- HellaSwag
- MMLU
- Llama-2-7B

Planned smoke target, if the dependency gate is resolved later:

- model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- task: PPL-only smoke
- quantization: `W4A4KV4`
- train batch size: `1`
- eval batch size: `1`

Would-be direct commands:

```bash
cd /mnt/e/SpinQuant-official
source /mnt/e/SpinQuant-official/.venv-spinquant/bin/activate

torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
  --input_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_rotation_path /mnt/e/SEQ_Clean/results/spinquant_official_debug/<timestamp>/rotations \
  --output_dir /mnt/e/SEQ_Clean/results/spinquant_official_debug/<timestamp>/logs/step1_output \
  --logging_dir /mnt/e/SEQ_Clean/results/spinquant_official_debug/<timestamp>/logs/step1_tb \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --log_on_each_node False \
  --per_device_train_batch_size 1 \
  --logging_steps 1 \
  --learning_rate 1.5 \
  --weight_decay 0. \
  --lr_scheduler_type cosine \
  --gradient_checkpointing True \
  --save_safetensors False \
  --max_steps 100 \
  --w_bits 4 \
  --a_bits 4 \
  --k_bits 4 \
  --v_bits 4 \
  --w_clip \
  --a_asym \
  --k_asym \
  --v_asym \
  --k_groupsize 128 \
  --v_groupsize 128
```

```bash
cd /mnt/e/SpinQuant-official
source /mnt/e/SpinQuant-official/.venv-spinquant/bin/activate

torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
  --input_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --do_train False \
  --do_eval True \
  --per_device_eval_batch_size 1 \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --save_safetensors False \
  --w_bits 4 \
  --a_bits 4 \
  --k_bits 4 \
  --v_bits 4 \
  --w_clip \
  --a_asym \
  --k_asym \
  --v_asym \
  --k_groupsize 128 \
  --v_groupsize 128 \
  --rotate \
  --optimized_rotation_path /mnt/e/SEQ_Clean/results/spinquant_official_debug/<timestamp>/rotations/R.bin
```

These commands were documented but not executed.

## Adapter Status

Adapter implementation status:

- not implemented

Files intentionally not created:

- `third_party_quant/adapters/spinquant_adapter.py`
- `third_party_quant/spinquant_compare.py`

Reason:

- direct official smoke did not pass
- upstream dependency installation did not complete cleanly
- the requested gating rule was to avoid integration if the official path could not be installed or smoke-tested safely

## Naming

If this method is implemented later, it should be named:

- `spinquant_official`

It should not be named:

- `spinquant_llmc`

Reason:

- this investigation targets the separate official upstream repository from `facebookresearch/SpinQuant`
- the stable LLMC checkout used by SEQ does not contain validated SpinQuant support
- keeping the name distinct avoids implying that SpinQuant is available through the stable LLMC integration path

## Recommendation

Current recommended Llama-2-7B method list remains unchanged:

- `base`
- `seq`
- `gptq_llmc`
- `smoothquant_llmc`
- `awq_llmc`
- `rtn_llmc`

Do not add `spinquant_official` yet.

Recommended next step before any SEQ adapter work:

- resolve a clean, reproducible `fast-hadamard-transform` install for the official upstream environment
- rerun the direct TinyLlama two-step smoke
- only after a successful direct PPL result, add a separate adapter path in SEQ
