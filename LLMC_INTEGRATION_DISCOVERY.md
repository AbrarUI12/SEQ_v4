# LLMC Integration Discovery

Date: 2026-05-12
Repo root audited: `E:\SEQ_Clean`

## A. Executive Summary

The clean SEQ checkout already has one working pattern for an external faithful baseline: `run_compare_matrix.py` dispatches `seq` and upstream `omniquant`, then normalizes lm-eval outputs into `results/compare_real__.../global_summary.csv`.

The safest LLMC integration is to extend that same compare runner, but keep all LLMC-specific logic in a new external adapter/orchestrator layer outside `seq_core`. That preserves SEQ core behavior and reuses the existing summary/output conventions.

Key findings:

- The current compare runner only supports `seq` and `omniquant`.
- The current SEQ core pipeline writes richer PPL/latency/memory/disk metrics than `run_compare_matrix.py` currently surfaces.
- `experiments.yaml` still contains internal baseline config blocks such as `ptq_w8a8`, `smoothquant_w8a8`, `gptq_w4a16`, and `rtn`, but those methods are not currently dispatchable in this clean matrix path.
- `third_party_quant/compare_methods.py` also only supports `omniquant`, and it is not wired into `seq_core.pipeline`.
- A local adjacent LLMC/LightCompress checkout was not found under `E:\SEQ_Clean`, `E:\`, or the user profile during this audit. LLMC findings below therefore combine local SEQ inspection with upstream LightCompress documentation/repo inspection.

Bottom line:

- Recommended implementation option: keep `run_compare_matrix.py` as the public entrypoint, add a thin LLMC adapter/orchestrator module, and do not modify `entropy_metrics.py`, `precision_policy.py`, `quantize_model.py`, or SEQ algorithm logic.
- Recommended first target methods: `gptq_llmc`, `smoothquant_llmc`, `awq_llmc`, `rtn_llmc`.
- `omniquant_llmc` is feasible, but best treated as a second phase because LLMC’s preferred OmniQuant flow is a two-step AWQ-plus-OmniQuant pipeline rather than a single standalone command.

## B. Current SEQ Benchmark Architecture

### Observed top-level structure

Present in this checkout:

- `run_compare_matrix.py`
- `seq_core/pipeline.py`
- `benchmarks/evaluation_suite.py`
- `benchmarks/core.py`
- `benchmarks/eval_config.py`
- `benchmarks/metrics_utils.py`
- `calibration_prompts.json`
- `eval_prompts.json`
- `experiments.yaml`
- `experiments.smoke.yaml`
- `requirements.clean.lock.txt`
- `requirements.lm_eval.txt`
- `third_party_quant/`

Expected in the request but not present in this clean checkout:

- root-level `pipeline.py`
- `run_full_eval.py`
- `baselines_ptq.py`
- root-level `evaluation_suite.py`
- root-level `benchmarks.py`
- `requirements.lock.txt`

So the current repository is a modularized layout, with the active evaluation stack under `seq_core/` and `benchmarks/`.

### How `run_compare_matrix.py` dispatches methods

`run_compare_matrix.py` is the current external comparison entrypoint.

Observed behavior:

- `SUPPORTED_METHODS = ("seq", "omniquant")`
- `METHOD_ALIASES = {"omniqunat": "omniquant"}`
- `main()` loops over `models` and `methods`
- `seq` dispatch path calls `_run_seq(...)`
- `omniquant` dispatch path calls `_run_omniquant(...)`
- after each method run, `_run_lm_eval_for_method(...)` is called
- outputs are flattened into `rows`
- `global_summary.json` and `global_summary.csv` are written under `results/compare_real__...`

### How `seq` is run

`_run_seq(...)`:

- builds a SEQ-specific compare config via `_make_seq_config(...)`
- disables local proxy benchmark tasks in that compare mode
- calls `seq_core.pipeline.run_experiment(...)`
- finds the new run directory under `.../seq/seq_runs/run_*`
- points `model_path` to `model_quantized/` if a reloadable `config.json` exists

### How the base and quant runs happen inside SEQ

Inside `seq_core/pipeline.py`, `run_experiment(...)` does:

1. Load baseline model/tokenizer.
2. Run `benchmarks.evaluation_suite.run_full_suite(...)` for baseline.
3. Write `eval_baseline/eval_summary.json`.
4. Write `bench_baseline/bench_summary.json`.
5. Run SEQ entropy collection and mixed-precision replacement.
6. Save quantized model to `model_quantized/`.
7. Run `run_full_suite(...)` on the quantized SEQ model.
8. Write `eval_quant/eval_summary.json`.
9. Write `bench_quant/bench_summary.json`.
10. Build markdown report and append `runs/index.csv`.

### How current PTQ methods are run

In this clean checkout, they mostly are not.

What exists:

- `experiments.yaml` contains `compare_methods` entries for:
  - `ptq_w8a8`
  - `smoothquant_w8a8`
  - `gptq_w4a16`
  - `rtn`
  - `zeroquant`
  - `omniquant`
  - `spinquant`
- `third_party_quant/compare_methods.py` exists
- `third_party_quant/compare_methods.py` supports only `omniquant`

What does not exist:

- no active dispatcher in `run_compare_matrix.py` for `ptq_w8a8`, `smoothquant_w8a8`, `gptq_w4a16`, or `rtn`
- no active support in `third_party_quant/compare_methods.py` for those methods
- no evidence that `seq_core.pipeline` invokes those methods

Conclusion:

- In this clean repo state, current internal PTQ baselines are configuration remnants or legacy scaffolding, not currently faithful external runners.

## C. Current SEQ Output Schema

There are two relevant output families.

### 1. Compare-matrix output schema

Produced by `run_compare_matrix.py`:

- `results/compare_real__<models>__<methods>__lm-eval-<tasks>__<timestamp>/metadata.json`
- per-model/per-method `summary.json`
- root `global_summary.json`
- root `global_summary.csv`

Observed `global_summary.csv` base columns:

- `model`
- `method`
- `status`
- `reason`
- `tasks`
- `model_path`
- `run_dir`

Additional dynamic columns:

- `lm_eval__status`
- `lm_eval__tasks`
- zero or more `lm_eval__<task>__<metric>` columns

This summary is currently lm-eval-centric and does not include PPL/latency/memory/disk columns.

### 2. SEQ run-level metric schema

Produced by `seq_core.pipeline.run_experiment(...)`:

- `eval_baseline/eval_summary.json`
- `eval_quant/eval_summary.json`
- `bench_baseline/bench_summary.json`
- `bench_quant/bench_summary.json`
- `runs/index.csv`

Observed `bench_summary.json` columns:

- `size`
- `ppl`
- `latency`
- `memory`
- `effective_bits_per_param`
- `disk_size_bytes`
- `disk_size_GB`
- `notes`

Observed `eval_summary.json` top-level keys:

- `tail_risk`
- `json_stress`
- `temperature_sweep`
- `long_context`
- `perplexity`
- `mmlu`
- `zero_shot`
- `size`
- `latency`
- `memory`
- `warnings`

Observed `runs/index.csv` columns:

- `run_id`
- `timestamp`
- `model_name`
- `experiment_name`
- `device`
- `dtype`
- `weight_high_pct`
- `act_high_pct`
- `effective_bits`
- `tail_exact_match`
- `json_success_rate`
- `latency_p50_sec`
- `tokens_per_sec_mean`
- `peak_memory_bytes`
- `run_dir`

## D. Current Internal Baseline Limitations

1. Dispatch limitation

- The live compare runner does not expose `ptq_w8a8`, `smoothquant_w8a8`, `gptq_w4a16`, or `rtn`.

2. Fidelity limitation

- For SEQ itself, quantization is applied in-process with bitsandbytes-backed modules.
- The repo documentation explicitly notes that quantized lm-eval is skipped by default unless a reloadable quantized path exists.
- That means some current “quantized” evaluation paths are still bounded by HF reloadability rather than by a backend-faithful external quantized runtime.

3. Architecture drift

- `experiments.yaml` still advertises internal PTQ method blocks, but the clean compare stack no longer dispatches them.
- `docs/LM_EVAL_INTEGRATION.md` says older compare entrypoints are not present, but `run_compare_matrix.py` is present now. This documentation is stale relative to the current tree.

4. External-method asymmetry

- The repo already has a clear external-adapter rule for OmniQuant, but no equivalent adapter path for LLMC.

## E. LLMC Repository Findings

### Local discovery result

No local adjacent LLMC/LightCompress checkout was found during this audit.

Not found locally under:

- `E:\`
- `E:\SEQ_Clean`
- `C:\Users\T2510647`

Local LLMC findings are therefore unavailable.

### Upstream project identity

Upstream project:

- GitHub repo: `ModelTC/LightCompress`
- Historical name: `LLMC`

Relevant upstream facts observed from docs/repo:

- LightCompress was formerly named LLMC.
- Main runner script: `scripts/run_llmc.sh`
- Main Python entrypoint: `llmc/__main__.py`
- Modified downstream evaluation helper: `scripts/run_lm_eval.sh`
- Supported quantization algorithms include:
  - AWQ
  - GPTQ
  - SmoothQuant
  - OmniQuant
  - RTN

### LLMC execution model

Observed flow from upstream:

- `scripts/run_llmc.sh` launches `torchrun`
- target is `llmc/__main__.py --config <config> --task_id <id>`
- configs live under `configs/quantization/...`
- algorithm outputs depend on `save.*` flags in config

### LLMC save/output modes observed

From upstream docs and `llmc/__main__.py`:

- `save_trans: True`
  - writes `transformed_model/`
  - intended for weight-transformed floating-point models
- `save_fake: True`
  - writes `fake_quant_model/`
  - pseudo-quant/fake-quant artifact
- `save_vllm: True`
  - writes `vllm_quant_model/`
  - backend-specific real quant export
- `save_autoawq: True`
  - writes `autoawq_quant_model/`
- `save_mlcllm: True`
  - writes `mlcllm_quant_model/`
- `save_lightllm: True`
  - writes `lightllm_quant_model/`
- `save_sgl: True`
  - writes `sgl_quant_model/`

### LLMC input model format

Observed requirement:

- LLMC expects the input model to already be in Hugging Face format.

### LLMC built-in eval modes

Observed concepts:

- `eval.eval_pos` can include:
  - `pretrain`
  - `transformed`
  - `fake_quant`
- LLMC performs internal PPL-style evaluation on datasets such as `wikitext2`, `c4`, and `ptb`
- LLMC also ships a separate `tools/llm_eval.py` and `scripts/run_lm_eval.sh`

Important caveat:

- `tools/llm_eval.py` logs a warning that it “only supports transformed/original model type”.
- That makes fake-quant and backend-specific artifact downstream evaluation a likely integration risk.

## F. Exact LLMC Commands Discovered

### Generic LLMC runner pattern

Upstream `scripts/run_llmc.sh` effectively does:

```bash
llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH
task_name=<task_name>
config=${llmc}/configs/quantization/.../<config>.yml

torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  --rdzv_id <random_port> \
  --rdzv_backend c10d \
  --rdzv_endpoint 127.0.0.1:<random_port> \
  ${llmc}/llmc/__main__.py \
  --config $config \
  --task_id <same_random_port>
```

### GPTQ W4A16

Method config:

- `configs/quantization/methods/GPTQ/gptq_w_only.yml`

Backend-real-quant VLLM config:

- `configs/quantization/backend/vllm/gptq_w4a16.yml`

Discovered method-level command pattern:

```bash
llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH
task_name=gptq_w4a16
config=${llmc}/configs/quantization/methods/GPTQ/gptq_w_only.yml
bash ${llmc}/scripts/run_llmc.sh
```

Discovered backend-export command pattern:

```bash
llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH
task_name=gptq_vllm_w4a16
config=${llmc}/configs/quantization/backend/vllm/gptq_w4a16.yml
bash ${llmc}/scripts/run_llmc.sh
```

### SmoothQuant W8A8

Method config:

- `configs/quantization/methods/SmoothQuant/smoothquant_w_a.yml`

Backend-real-quant VLLM config:

- `configs/quantization/backend/vllm/smoothquant_w8a8.yml`

Discovered command pattern:

```bash
llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH
task_name=smoothquant_w8a8
config=${llmc}/configs/quantization/methods/SmoothQuant/smoothquant_w_a.yml
bash ${llmc}/scripts/run_llmc.sh
```

### AWQ W4A16

Method config:

- `configs/quantization/methods/Awq/awq_w_only.yml`

Backend-real-quant VLLM config:

- `configs/quantization/backend/vllm/awq_w4a16.yml`

Discovered command pattern:

```bash
llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH
task_name=awq_w4a16
config=${llmc}/configs/quantization/methods/Awq/awq_w_only.yml
bash ${llmc}/scripts/run_llmc.sh
```

### RTN W4A16 / W8A16

Method config found:

- `configs/quantization/methods/RTN/rtn_w_only.yml` with default 4-bit weight-only settings

Backend-real-quant VLLM config found:

- `configs/quantization/backend/vllm/rtn_w8a16.yml`

Discovered command pattern:

```bash
llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH
task_name=rtn_w_only
config=${llmc}/configs/quantization/methods/RTN/rtn_w_only.yml
bash ${llmc}/scripts/run_llmc.sh
```

Preferred faithful backend-export pattern:

```bash
llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH
task_name=rtn_vllm_w8a16
config=${llmc}/configs/quantization/backend/vllm/rtn_w8a16.yml
bash ${llmc}/scripts/run_llmc.sh
```

### OmniQuant W4A16

Observed LLMC best-practice path is a two-step combination flow, not a single simple one-step method config.

Step 1 config:

- `configs/quantization/combination/awq_comb_omni/w4a16g128/step_1_awq.yml`

Step 2 config:

- `configs/quantization/combination/awq_comb_omni/w4a16g128/step_2_omniq.yml`

Discovered command pattern:

```bash
llmc=/path/to/LightCompress
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=awq_omni_step1_awq
config=${llmc}/configs/quantization/combination/awq_comb_omni/w4a16g128/step_1_awq.yml
bash ${llmc}/scripts/run_llmc.sh

task_name=awq_omni_step2_omniq
config=${llmc}/configs/quantization/combination/awq_comb_omni/w4a16g128/step_2_omniq.yml
bash ${llmc}/scripts/run_llmc.sh
```

Interpretation:

- `omniquant_llmc` is feasible.
- It should probably be explicitly labeled as an LLMC OmniQuant variant that follows LLMC’s AWQ-initialized OmniQuant recipe.

### LLMC downstream lm-eval command pattern

Observed upstream helper:

```bash
accelerate launch --multi_gpu --num_processes 4 llmc/tools/llm_eval.py \
  --config llmc/configs/quantization/RTN/rtn_quarot.yml \
  --model hf \
  --quarot \
  --tasks lambada_openai,arc_easy \
  --model_args parallelize=False,trust_remote_code=True \
  --batch_size 64 \
  --output_path ./save/lm_eval \
  --log_samples
```

Practical implication:

- LLMC has a downstream lm-eval path, but it is not a drop-in replacement for SEQ’s current evaluation stack.
- It is best treated as a fallback, not as the primary canonical evaluation path for SEQ integration.

## G. Recommended Integration Architecture

### Recommended option

Extend `run_compare_matrix.py`, but move all LLMC-specific logic into a new external baseline layer.

Recommended shape:

1. `run_compare_matrix.py` remains the public comparison entrypoint.
2. Add a new LLMC adapter/orchestrator module under `third_party_quant/`.
3. Do not route LLMC through `seq_core.pipeline`.
4. Do not modify SEQ entropy or quantization core.

### Why this is the safest option

- It matches the existing OmniQuant external-integration pattern.
- It keeps SEQ and LLMC concerns separated.
- It preserves current output root layout under `results/compare_real__...`.
- It minimizes risk to `seq_core`.

### Proposed execution flow

For each `*_llmc` method:

1. Resolve LLMC repo path and Python executable.
2. Materialize an LLMC config for the requested method/model/calibration settings.
3. Launch LLMC externally.
4. Detect produced artifact type:
   - `transformed_model`
   - `fake_quant_model`
   - `vllm_quant_model`
   - `autoawq_quant_model`
   - etc.
5. Decide evaluation path:
   - If a Hugging Face-loadable artifact is produced and verified, run SEQ `run_full_suite(...)` and `run_lm_eval_suite(...)`.
   - Else, record backend artifact paths, disk size, LLMC internal eval if available, and mark latency/PPL comparability limits clearly.
6. Normalize per-method summary into the same compare result row style as current `run_compare_matrix.py`.

### Canonical evaluation recommendation

Use SEQ evaluation as the canonical aggregator whenever the LLMC output is HF-loadable.

Recommended preference order:

1. `run_full_suite(...)` for:
   - PPL
   - latency/tokens_per_sec
   - memory
   - JSON/tail/long-context if desired
2. `run_lm_eval_suite(...)` for HellaSwag/ARC/PIQA summary flattening
3. LLMC’s own lm-eval helper only as fallback

Reason:

- This preserves fairness and keeps metric definitions aligned with SEQ.

### Important policy recommendation

For phase 1, treat backend-specific real-quant latency as optional and not directly comparable to SEQ HF/bitsandbytes latency unless it is measured under a matched benchmark harness and clearly labeled.

## H. Files To Add

Recommended new files:

- `third_party_quant/adapters/llmc_adapter.py`
  - subprocess launcher
  - provenance writer
  - LLMC repo/env/config resolution
  - dry-run support

- `third_party_quant/llmc_compare.py`
  - method-specific orchestration
  - artifact discovery
  - summary normalization
  - evaluation-path selection

- `third_party_quant/llmc_templates/`
  - `gptq_w4a16.yml`
  - `smoothquant_w8a8.yml`
  - `awq_w4a16.yml`
  - `rtn_w8a16.yml`
  - `awq_omni_step1_awq.yml`
  - `awq_omni_step2_omniq.yml`

- `third_party_quant/docs/llmc_validation.md`
  - equivalent to current `omniquant_validation.md`

Optional but useful:

- `third_party_quant/envs/llmc-upstream.environment.yml`

## I. Files To Minimally Modify

Recommended minimal modifications:

- `run_compare_matrix.py`
  - add new supported methods
  - add LLMC CLI flags
  - call LLMC orchestration module
  - optionally enrich compare rows with extra normalized columns

- `experiments.yaml`
  - add `compare_methods` blocks for:
    - `gptq_llmc`
    - `smoothquant_llmc`
    - `awq_llmc`
    - `rtn_llmc`
    - `omniquant_llmc`

- `experiments.smoke.yaml`
  - add minimal smoke defaults for one or two LLMC methods

- `benchmarks/README.md`
  - update compare-runner examples and method list

Do not modify unless absolutely necessary:

- `seq_core/entropy_metrics.py`
- `seq_core/precision_policy.py`
- `seq_core/quantize_model.py`

## J. Metrics Mapping Table

| LLMC metric or artifact | Preferred source in integration | SEQ compare/global field |
| --- | --- | --- |
| model identifier | request/config | `model` |
| method label | orchestration layer | `method` |
| run status | adapter/orchestrator | `status` |
| failure reason | adapter/orchestrator | `reason` |
| lm-eval task list | SEQ `run_lm_eval_suite` or fallback LLMC helper | `tasks`, `lm_eval__tasks` |
| lm-eval task metric | SEQ normalized lm-eval summary | `lm_eval__<task>__<metric>` |
| reloadable artifact path | artifact discovery | `model_path` |
| method run directory | orchestration layer | `run_dir` |
| perplexity | SEQ `eval_summary.json -> perplexity.ppl` if HF-loadable; else parsed LLMC eval if trustworthy | proposed extra column `ppl` |
| PPL dataset name | SEQ eval or LLMC config | proposed extra column `ppl_dataset` |
| PPL seq len | SEQ eval or LLMC config | proposed extra column `ppl_seq_len` |
| quantized disk bytes | filesystem size of saved artifact | proposed extra column `quant_disk_bytes` |
| quantized disk GB | filesystem size of saved artifact | proposed extra column `quant_disk_gb` |
| latency prefill p50 | SEQ bench summary if same HF path; else null | proposed extra column `prefill_p50_ms` |
| decode throughput | SEQ bench summary if same HF path; else null | proposed extra column `tokens_per_sec` |
| peak memory | SEQ eval/bench summary if same HF path; else null | proposed extra column `peak_mem_gb` |
| artifact kind | adapter/orchestrator | proposed extra column `artifact_kind` |
| backend kind | adapter/orchestrator | proposed extra column `backend` |
| comparability note | adapter/orchestrator | proposed extra column `notes` |

Recommendation:

- keep current row fields unchanged
- add extra columns rather than replacing anything
- allow blank/null for non-comparable metrics

## K. Proposed Method Names

Use exactly:

- `gptq_llmc`
- `smoothquant_llmc`
- `awq_llmc`
- `rtn_llmc`
- `omniquant_llmc`

Keep existing simulated/internal names unchanged:

- `ptq_w8a8`
- `smoothquant_w8a8`
- `gptq_w4a16`

## L. Proposed Folder Layout

```text
third_party_quant/
  adapters/
    omniquant_adapter.py
    llmc_adapter.py                  # new
  llmc_compare.py                    # new
  llmc_templates/                    # new
    gptq_w4a16.yml
    smoothquant_w8a8.yml
    awq_w4a16.yml
    rtn_w8a16.yml
    awq_omni_step1_awq.yml
    awq_omni_step2_omniq.yml
  docs/
    omniquant_validation.md
    llmc_validation.md               # new
  envs/
    omniquant-upstream.environment.yml
    llmc-upstream.environment.yml    # optional new
```

Result layout recommendation:

```text
results/compare_real__<models>__<methods>__lm-eval-<tasks>__<timestamp>/
  metadata.json
  global_summary.json
  global_summary.csv
  <model_slug>/
    gptq_llmc/
      requested_config.json
      llmc_adapter_summary.json
      summary.json
      llmc_run/
      eval/
      bench/
    smoothquant_llmc/
    awq_llmc/
    rtn_llmc/
    omniquant_llmc/
```

## M. Smoke-Test Plan

### Smoke test 0: discovery sanity

Goal:

- verify LLMC repo path is detected
- verify config rendering
- verify subprocess command construction
- do not run full quantization yet

Needed capability:

- `--llmc_dry_run`

### Smoke test 1: smallest real method

Recommended first real method:

- `gptq_llmc`

Recommended smallest model:

- `facebook/opt-125m`

Recommended eval scope:

- lm-eval: `hellaswag` only
- limit: `10`
- calibration samples: `8` or `16`
- PPL seq len: use smoke config values

### Smoke test 2: transformed-model path

Method:

- `smoothquant_llmc`

Reason:

- easiest candidate for `save_trans` plus SEQ-side reload/eval

### Smoke test 3: backend-real-quant path

Method:

- `rtn_llmc` with VLLM export

Reason:

- simplest real-quant backend export
- good for artifact discovery and disk-size checks

## N. Full Benchmark Plan

Phase 1:

- implement adapter/orchestrator
- support `gptq_llmc`, `smoothquant_llmc`, `awq_llmc`, `rtn_llmc`
- keep evaluation canonical in SEQ when artifact reload is confirmed
- write compare rows with nulls for non-comparable latency fields

Phase 2:

- implement `omniquant_llmc`
- support two-step AWQ-plus-Omni LLMC flow
- add explicit provenance that this is LLMC’s OmniQuant recipe

Phase 3:

- verify which LLMC artifact modes are Hugging Face reloadable in practice
- only after that, decide whether to expose:
  - PPL
  - latency
  - peak memory
  - long-context
  - JSON/tail-risk

Phase 4:

- add fair backend-specific latency mode if needed
- only compare against SEQ latency if harness, warmup, measured runs, tokenizer path, and runtime backend are aligned

## O. Risks And Mitigations

### 1. LLMC repo not present locally

Risk:

- implementation cannot be validated against a local checkout yet

Mitigation:

- require explicit `--llmc_repo` path
- support a dry-run mode first

### 2. Fake-quant artifact may not be HF-loadable

Risk:

- SEQ `run_full_suite(...)` may not be able to load `fake_quant_model`

Mitigation:

- detect artifact kind explicitly
- only run SEQ eval on verified reloadable artifacts
- otherwise write nulls plus notes

### 3. `save_trans` is not “real quant”

Risk:

- transformed FP weights are useful for accuracy experiments but are not a backend-faithful quantized runtime

Mitigation:

- label artifact kind clearly
- distinguish `transformed_hf` from `vllm_quant` or `autoawq_quant`

### 4. Backend latency is not apples-to-apples with SEQ HF latency

Risk:

- vLLM/AutoAWQ latency will not match SEQ’s current HF/bitsandbytes latency harness

Mitigation:

- do not merge backend latency into canonical latency columns unless the harness is aligned
- store separate backend-latency notes or fields

### 5. Calibration mismatch

Risk:

- LLMC configs default to `pileval` or `wikitext2` preprocessing that may not match SEQ prompt-list calibration

Mitigation:

- render LLMC configs from SEQ experiment values
- explicitly set:
  - calibration dataset
  - calibration sample count
  - seq len
  - seeds

### 6. PPL mismatch

Risk:

- SEQ uses configurable proxy/canonical PPL logic
- LLMC internal eval uses its own PPL flow

Mitigation:

- prefer SEQ PPL evaluation when the LLMC artifact can be reloaded
- otherwise mark PPL source as `llmc_internal`

### 7. Windows / WSL / path issues

Risk:

- LLMC runner is shell-script and torchrun oriented
- path quoting and environment setup can differ

Mitigation:

- adapter should invoke Python entrypoint directly when possible
- normalize all paths in Python

### 8. Dependency drift

Risk:

- CUDA, torch, vLLM, AutoAWQ, bitsandbytes, and transformers versions may diverge between SEQ and LLMC envs

Mitigation:

- use a separate LLMC environment
- record versions and provenance per run

### 9. OmniQuant naming ambiguity

Risk:

- `omniquant_llmc` in LLMC may mean AWQ-initialized OmniQuant, not raw upstream OpenGVLab OmniQuant

Mitigation:

- state provenance explicitly in summary notes
- keep current upstream `omniquant` baseline separate

## P. Open Questions That Block Implementation

1. Where will the local LightCompress checkout live?

- Proposed default: `..\LightCompress`
- Currently not present in the audited environment

2. Which LLMC artifact type should be canonical for each method?

- `save_trans`
- `save_fake`
- `save_vllm`
- `save_autoawq`

This needs practical validation per method.

3. Do you want backend latency in the same summary now, or only after a matched backend harness is added?

My recommendation: defer backend latency comparability to phase 2.

4. Should `omniquant_llmc` mean:

- raw LLMC OmniQuant method if a stable single-step config exists, or
- LLMC’s documented AWQ-plus-OmniQuant best-practice recipe?

My recommendation: use the documented AWQ-plus-OmniQuant recipe and say so explicitly.

5. Should LLMC methods be added directly to `run_compare_matrix.py`, or should there also be a standalone `run_llmc_baselines.py` for debugging?

My recommendation: direct compare-runner integration first, standalone script only if debug friction appears.

## External References Consulted

Local SEQ files:

- `run_compare_matrix.py`
- `seq_core/pipeline.py`
- `benchmarks/evaluation_suite.py`
- `benchmarks/core.py`
- `benchmarks/eval_config.py`
- `third_party_quant/adapters/omniquant_adapter.py`
- `third_party_quant/compare_methods.py`
- `experiments.yaml`
- `experiments.smoke.yaml`

Upstream LLMC/LightCompress references used because no local checkout was found:

- GitHub repo: `https://github.com/ModelTC/LightCompress`
- Docs root: `https://llmc-en.readthedocs.io/en/latest/`
- Quickstart: `https://llmc-en.readthedocs.io/en/stable/quickstart.html`
- Config docs: `https://llmc-en.readthedocs.io/en/stable/configs.html`
- AWQ docs: `https://llmc-en.readthedocs.io/en/stable/practice/awq.html`
- AWQ + OmniQuant docs: `https://llmc-en.readthedocs.io/en/stable/practice/awq_omni.html`
- VLLM backend docs: `https://llmc-en.readthedocs.io/en/stable/backend/vllm.html`
- AutoAWQ backend docs: `https://llmc-en.readthedocs.io/en/stable/backend/autoawq.html`

